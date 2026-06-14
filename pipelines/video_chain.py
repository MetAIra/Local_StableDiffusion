"""シーン連鎖動画生成モジュール

「LTX-Video の i2v チェーン + ffmpeg xfade 連結」で最長10分級の長尺動画を作る。

設計:
- 各シーン = (prompt, num_frames, guidance, seed_offset, fresh_start) のリスト
- fresh_start=True か先頭シーンは t2v、それ以外は前シーンの最終フレームを image= に渡して i2v
- 全シーン生成後、ffmpeg xfade で滑らかに連結（既定 0.3 秒クロスフェード）
- 出力フォルダは1個に統一: scene_NNN.mp4 + _final_concat.mp4 + metadata.csv
- is_stop_requested() を毎シーンチェックして途中中断対応（既に出来た分は連結する）

LTX-Video 限定の理由:
- 本リポジトリで「画像 + プロンプト両対応」の唯一の動画モデル
- SVD-XT は画像のみ（プロンプト無視）、AnimateDiff/Wan2.1/CogVideoX は t2v 専用
"""
import os
import gc
import traceback
from typing import Optional, List, Tuple

import torch

from config import (
    VIDEO_MODELS, VIDEO_DEFAULT_FORMAT, VIDEO_OUTPUT_DIR_PREFIX,
    is_stop_requested, clear_stop,
)
from utils.batch_parsing import save_metadata_csv
from utils.file import create_output_dir
from utils.video_interpolation import extract_last_frame, concat_videos_xfade
from .video_gen import generate_video
from .video_manager import video_pipeline_manager

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# シーン上限（暴走防止）。10 分 = 約 120 シーン × 5 秒想定なので 200 で十分安全側
VIDEO_CHAIN_MAX_SCENES = 200

# 連鎖生成に対応するモデル（プロンプト + 画像入力ができるモデル）
VIDEO_CHAIN_SUPPORTED_MODELS = {"ltx_video"}


def parse_scenes_text(text: str) -> List[dict]:
    """テキスト形式のシーン定義をパース

    1 行 1 シーン。タブ区切りで以下のフィールド:
        prompt [TAB] num_frames [TAB] guidance [TAB] seed_offset [TAB] fresh_start(0/1)

    末尾フィールドは省略可。`#` で始まる行はコメント。空行はスキップ。
    プロンプトには TAB を含めない（含む場合は事前にスペースに置換）。
    """
    scenes: List[dict] = []
    for line in (text or "").splitlines():
        s = line.rstrip()
        if not s.strip():
            continue
        if s.lstrip().startswith("#"):
            continue
        cols = s.split("\t")
        prompt = cols[0].strip()
        if not prompt:
            continue

        def _intish(idx, default):
            if idx >= len(cols):
                return default
            v = cols[idx].strip()
            try:
                return int(v) if v else default
            except ValueError:
                return default

        def _floatish(idx, default):
            if idx >= len(cols):
                return default
            v = cols[idx].strip()
            try:
                return float(v) if v else default
            except ValueError:
                return default

        scenes.append({
            "prompt": prompt,
            "num_frames": _intish(1, 0),  # 0 はデフォルト採用
            "guidance": _floatish(2, 0.0),
            "seed_offset": _intish(3, len(scenes)),  # 既定はインデックス
            "fresh_start": bool(_intish(4, 0)),
        })
    return scenes


def generate_video_chain(
    scenes: List[dict],
    model_key: str = "ltx_video",
    negative_prompt: str = "",
    width: int = 704,
    height: int = 480,
    fps: int = 24,
    default_num_frames: int = 121,
    default_guidance: float = 3.0,
    num_inference_steps: int = 30,
    base_seed: int = -1,
    crossfade_seconds: float = 0.3,
    xfade_transition: str = "fade",
    fmt: str = VIDEO_DEFAULT_FORMAT,
    smooth_target_fps: Optional[int] = None,
    smooth_method: str = "rife",
    smooth_mode: str = "mci",
    custom_base_path: Optional[str] = None,
    save_lastframes: bool = True,
) -> Tuple[List[str], str]:
    """シーンリストを順次生成し、xfade で1本に連結

    Args:
        scenes: シーン定義のリスト。各要素は
            { "prompt": str, "num_frames": int (0=デフォルト), "guidance": float (0=デフォルト),
              "seed_offset": int, "fresh_start": bool }
        model_key: 現状 "ltx_video" のみサポート
        width / height: 全シーン共通の解像度
        fps: 元生成 FPS（補間前）
        default_num_frames / default_guidance: シーン側で省略時に使う既定値
        base_seed: -1 ならランダム、整数なら base_seed + scene.seed_offset
        crossfade_seconds: xfade の長さ。0 にすれば xfade スキップで単純結合
        smooth_target_fps: 各シーン生成後に RIFE/minterpolate でこの FPS に補間
        save_lastframes: True なら境界フレームを jpg で残す（デバッグ・確認用）

    Returns:
        (生成ファイル一覧 [連結後mp4 含む], ステータスメッセージ)
    """
    if model_key not in VIDEO_CHAIN_SUPPORTED_MODELS:
        return [], (
            f"連鎖生成は現状 {sorted(VIDEO_CHAIN_SUPPORTED_MODELS)} のみ対応です "
            f"（{model_key} は不可）"
        )

    cfg = VIDEO_MODELS.get(model_key)
    if not cfg:
        return [], f"未知のモデル: {model_key}"

    if not scenes:
        return [], "シーンが空です"
    if len(scenes) > VIDEO_CHAIN_MAX_SCENES:
        return [], f"シーン数が多すぎます: {len(scenes)} > {VIDEO_CHAIN_MAX_SCENES}"

    # シーン妥当性の最低限チェック
    for i, s in enumerate(scenes):
        if not s.get("prompt", "").strip():
            return [], f"シーン {i} のプロンプトが空です"

    label_prompt = scenes[0]["prompt"][:30]
    out_dir = create_output_dir(
        _PROJECT_ROOT,
        f"{VIDEO_OUTPUT_DIR_PREFIX}_chain_{model_key}",
        label_prompt,
        category="video",
    )

    print(f"[Chain] {cfg['label']}  scenes={len(scenes)}  res={width}x{height}  fps={fps}  xfade={crossfade_seconds}s")

    scene_paths: List[str] = []
    csv_rows: List[dict] = []
    generated = 0
    stopped = False
    clear_stop()

    prev_last_frame = None  # PIL.Image
    base_seed_int = int(base_seed)

    for i, scene in enumerate(scenes):
        if is_stop_requested():
            stopped = True
            break

        prompt = scene["prompt"].strip()
        n_frames = int(scene.get("num_frames") or 0) or int(default_num_frames)
        guidance = float(scene.get("guidance") or 0.0) or float(default_guidance)
        seed_offset = int(scene.get("seed_offset", i))
        fresh_start = bool(scene.get("fresh_start", False))

        # シード: ベース -1 ならシーンごとにランダム、それ以外なら base + offset
        if base_seed_int < 0:
            cur_seed = -1
        else:
            cur_seed = base_seed_int + seed_offset

        # i2v 入力画像: 先頭シーン or fresh_start のときは None で t2v
        image_in = None if (i == 0 or fresh_start or prev_last_frame is None) else prev_last_frame

        prefix = f"scene_{i:03d}"
        print(f"  [{i+1}/{len(scenes)}] {'t2v' if image_in is None else 'i2v'}  "
              f"frames={n_frames}  guidance={guidance}  seed={cur_seed}  prompt={prompt[:60]!r}")

        try:
            out_path, msg = generate_video(
                model_key=model_key,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_in,
                num_frames=n_frames,
                fps=fps,
                width=int(width),
                height=int(height),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance),
                seed=cur_seed,
                fmt=fmt,
                smooth_target_fps=smooth_target_fps,
                smooth_method=smooth_method,
                smooth_mode=smooth_mode,
                custom_base_path=custom_base_path,
                output_dir=out_dir,
                filename_prefix=prefix,
            )
        except Exception as e:
            traceback.print_exc()
            return scene_paths, f"シーン {i} 生成中にエラー: {e}\n（{generated}/{len(scenes)}件完了）"

        if not (out_path and os.path.exists(out_path)):
            print(f"    FAIL: {msg[:200]}")
            # 1 シーン失敗で全停止（チェーンが分断するので後続も実質意味なし）
            return scene_paths, (
                f"シーン {i} の生成に失敗しました: {msg[:300]}\n"
                f"（{generated}/{len(scenes)}件完了。出力先: {out_dir}）"
            )

        scene_paths.append(out_path)
        generated += 1

        # 次シーンへ受け渡す最終フレーム
        # smooth 補間がかかっているとファイル名が変わっている可能性があるが、
        # out_path はその新ファイルなので extract_last_frame に渡せば OK
        lastframe_save = (
            os.path.join(out_dir, f"{prefix}_lastframe.jpg") if save_lastframes else None
        )
        prev_last_frame = extract_last_frame(out_path, lastframe_save)
        if prev_last_frame is None:
            print(f"    [WARN] 最終フレーム抽出失敗 -> 次シーンは fresh_start 扱い")

        csv_rows.append({
            "scene_idx": i,
            "filename": os.path.basename(out_path),
            "mode": "t2v" if image_in is None else "i2v",
            "prompt": prompt,
            "num_frames": n_frames,
            "guidance": guidance,
            "seed": cur_seed,
            "fresh_start": fresh_start,
            "width": width,
            "height": height,
            "fps": fps,
            "smooth_target_fps": smooth_target_fps,
            "smooth_method": smooth_method,
        })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # CSV
    csv_path = os.path.join(out_dir, "metadata.csv")
    save_metadata_csv(csv_path, csv_rows)

    # 連結（中断時もできた分だけ連結する）
    final_path: Optional[str] = None
    if len(scene_paths) >= 1:
        final_path = os.path.join(out_dir, "_final_concat.mp4")
        if crossfade_seconds and crossfade_seconds > 0 and len(scene_paths) >= 2:
            # GPU を空けてから ffmpeg を走らせる（メモリ競合回避）
            try:
                video_pipeline_manager.unload()
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[Chain] xfade 連結開始: {len(scene_paths)}本 / xfade={crossfade_seconds}s")
            final_path = concat_videos_xfade(
                scene_paths, final_path,
                crossfade_seconds=float(crossfade_seconds),
                transition=xfade_transition,
                target_fps=smooth_target_fps,
            )
        elif len(scene_paths) == 1:
            # 1本だけのときはそれをそのまま最終扱い
            final_path = scene_paths[0]
        else:
            final_path = None

    head = "中止" if stopped else "完了"
    msg_lines = [
        f"連鎖生成 {head}（{generated}/{len(scenes)}シーン）",
        f"モデル: {cfg['label']}",
        f"解像度: {width}x{height} / fps={fps}"
        + (f" -> {smooth_target_fps}fps 補間" if smooth_target_fps and smooth_target_fps > fps else ""),
        f"クロスフェード: {crossfade_seconds}s",
        f"CSV: {csv_path}",
        f"保存先: {out_dir}",
    ]
    if final_path and os.path.exists(final_path):
        msg_lines.append(f"連結後: {final_path}")
    return (scene_paths + ([final_path] if final_path and final_path not in scene_paths else [])), "\n".join(msg_lines)


# UI から使うプリセット
# 形式: name -> scenes_text（タブ区切り、1行1シーン）
# スタイル指示は各シーンプロンプトに inline で含める（共通前置は廃止）。
# 連鎖の i2v 引き継ぎを活かすために、隣接シーンは同じ被写体/世界観に揃え、
# 場面転換したい所には fresh_start=1 を立てる。
VIDEO_CHAIN_EXAMPLES = {
    # ========================================================================
    # 風景・場面物
    # ========================================================================
    "[風景] 季節の移り変わり 6シーン": (
        "cherry blossoms falling in a quiet japanese garden, soft pink petals, spring morning, cinematic landscape, masterpiece, ultra detailed\t97\t3.0\t0\t1\n"
        "lush green forest in summer, sunlight rays through leaves, slow camera dolly, cinematic, ultra detailed\t97\t3.0\t1\t0\n"
        "autumn maple trees with red and orange leaves, gentle wind, golden hour, cinematic landscape\t97\t3.0\t2\t1\n"
        "snowy mountain peaks under winter sky, snowflakes drifting, cold blue tones, cinematic\t97\t3.0\t3\t1\n"
        "frozen lake reflecting aurora borealis, night sky, magical atmosphere, ultra detailed\t97\t3.0\t4\t1\n"
        "first sunrise over snow-covered field, warm golden light, peaceful, cinematic landscape\t97\t3.0\t5\t1\n"
    ),
    "[風景] 都市 時間帯 4シーン": (
        "modern tokyo street at dawn, soft pink sky, empty road, cinematic, ultra realistic\t97\t3.0\t0\t1\n"
        "tokyo crossing at noon, crowd of people, bright daylight, cinematic, ultra realistic\t97\t3.0\t1\t0\n"
        "tokyo skyline at sunset, golden hour, dramatic clouds, cinematic\t97\t3.0\t2\t1\n"
        "neon-lit tokyo street at midnight, rain reflections, cyberpunk vibe, cinematic, atmospheric\t97\t3.0\t3\t1\n"
    ),
    "[風景] 海 波打ち際 5シーン": (
        "calm ocean waves rolling onto a white sandy beach, morning light, cinematic seascape, photorealistic\t97\t3.0\t0\t1\n"
        "seagulls flying over crashing waves, foam and spray, cinematic, slow motion\t97\t3.0\t1\t0\n"
        "underwater view of sunlight beams piercing blue ocean, cinematic, photorealistic\t97\t3.0\t2\t1\n"
        "coral reef with colorful fish swimming, slow motion, cinematic\t97\t3.0\t3\t0\n"
        "sunset reflection on the sea surface, warm orange glow, cinematic seascape\t97\t3.0\t4\t1\n"
    ),
    "[風景] 桜並木トンネル 4シーン": (
        "wide shot of a long cherry blossom tunnel in spring, pink petals raining down, soft sunlight, cinematic, ultra detailed\t97\t3.0\t0\t1\n"
        "slow forward dolly through the cherry blossom tunnel, petals swirling in the air, cinematic, peaceful\t97\t3.0\t1\t0\n"
        "deeper inside the tunnel, sunlight piercing through dense pink canopy, dappled light on the path, cinematic\t97\t3.0\t2\t0\n"
        "exit of the cherry blossom tunnel opening to a sunlit field, warm spring afternoon, cinematic landscape\t97\t3.0\t3\t0\n"
    ),
    "[風景] 宇宙旅行 5シーン": (
        "earth seen from low orbit, sunrise crescent over the planet, cinematic space view, ultra detailed\t97\t3.5\t0\t1\n"
        "spacecraft drifting past the moon toward the dark void, cinematic, photorealistic, slow motion\t97\t3.5\t1\t0\n"
        "warp speed effect, stars stretching into lines, dramatic light streaks, cinematic sci-fi\t97\t3.5\t2\t1\n"
        "spiral galaxy slowly rotating, nebula clouds in deep purple and blue, cosmic, cinematic\t97\t3.5\t3\t1\n"
        "supermassive black hole with glowing accretion disk, gravitational lensing, awe-inspiring, cinematic\t97\t3.5\t4\t1\n"
    ),
    "[風景] 雨上がりの街 4シーン": (
        "narrow city alley as rain stops, last drops falling, wet asphalt reflecting neon signs, cinematic atmospheric\t97\t3.0\t0\t1\n"
        "close-up of a puddle reflecting colorful neon lights, water rippling gently, cinematic, photorealistic\t73\t3.0\t1\t0\n"
        "water droplets falling from street tree leaves in slow motion, soft afternoon light returning, cinematic\t73\t3.0\t2\t0\n"
        "wide shot of the street as a rainbow appears over wet rooftops, cinematic urban landscape, peaceful\t97\t3.0\t3\t0\n"
    ),
    "[風景] 砂漠オアシス 5シーン": (
        "vast sand dunes at sunrise, long shadows, gentle wind blowing sand, cinematic desert landscape\t97\t3.0\t0\t1\n"
        "camel caravan crossing the dunes in slow motion, golden light, cinematic wide shot, photorealistic\t97\t3.0\t1\t0\n"
        "discovery of a hidden oasis between dunes, palm trees and clear blue water, cinematic, awe\t97\t3.0\t2\t0\n"
        "close-up of clear oasis water with ripples, palm reflections, slow motion droplet, cinematic\t73\t3.0\t3\t0\n"
        "desert at sunset, silhouette of palm trees against orange sky, cinematic landscape\t97\t3.0\t4\t1\n"
    ),
    "[風景] 北欧フィヨルド 4シーン": (
        "morning mist over a long norwegian fjord, dark mountains on both sides, cinematic, ultra detailed\t97\t3.0\t0\t1\n"
        "wide aerial shot of the fjord with a tiny boat far below, dramatic scale, cinematic landscape\t97\t3.0\t1\t0\n"
        "small red wooden cabin by the fjord shore, soft fog, peaceful, cinematic, photorealistic\t97\t3.0\t2\t0\n"
        "same fjord at night, aurora borealis dancing across the sky, reflection on still water, cinematic, magical\t97\t3.0\t3\t1\n"
    ),
    "[風景] 古代遺跡探訪 5シーン": (
        "explorer's wide view of an ancient stone temple half swallowed by jungle, sunlight through vines, cinematic, ultra detailed\t97\t3.5\t0\t1\n"
        "slow approach to the temple entrance flanked by huge weathered stone statues, dramatic shadows, cinematic\t97\t3.5\t1\t0\n"
        "interior corridor of the temple, dust motes in golden light beams, ancient wall carvings, cinematic\t97\t3.5\t2\t0\n"
        "central chamber with a glowing artifact on a stone altar, mysterious blue light, cinematic, atmospheric\t97\t3.5\t3\t0\n"
        "temple at sunset, mysterious light beams shooting up into the sky, epic fantasy, cinematic\t97\t3.5\t4\t1\n"
    ),

    # ========================================================================
    # キャラクター
    # （キャラ識別子の一貫性は LTX-Video の i2v 引き継ぎ依存。3〜4シーン以降は
    #   髪色・顔がドリフトしやすいので、ドリフトが気になったら fresh_start=1 を
    #   挟んで「別人にする」演出として使うのが現実的）
    # ========================================================================
    "[キャラ] アニメ少女 1日の流れ 5シーン": (
        "anime style girl with long pink hair and blue eyes wearing a white summer dress, waking up in her bedroom, soft morning light through curtains, cel-shaded, masterpiece, ultra detailed\t97\t3.5\t0\t1\n"
        "anime style girl with pink hair walking through a sunny city street with a cafe in background, holding a coffee cup, cel-shaded, cinematic\t97\t3.5\t1\t0\n"
        "anime style girl reading a book at a park bench under cherry blossoms, gentle breeze moving petals, cel-shaded, peaceful afternoon\t97\t3.5\t2\t0\n"
        "anime style girl with pink hair at a sunset rooftop overlooking the city, wind in her hair, dramatic warm light, cel-shaded\t97\t3.5\t3\t0\n"
        "anime style girl in her bedroom at night, lamp glowing softly, stars visible through window, cel-shaded, calm\t97\t3.5\t4\t0\n"
    ),
    "[キャラ] サイバーパンク戦士 アクション 6シーン": (
        "anime cyberpunk female warrior with silver hair and glowing cyan visor, black tech suit, standing in a neon-lit alley at night, rain falling, cinematic, dramatic\t97\t4.0\t0\t1\n"
        "same cyberpunk warrior drawing a glowing energy blade, sparks flying, motion blur, neon reflections on wet ground, cinematic action\t97\t4.0\t1\t0\n"
        "cyberpunk warrior running across a rooftop, neon city skyline behind, slow motion jump, dramatic camera angle, cinematic\t97\t4.0\t2\t0\n"
        "cyberpunk warrior fighting in a holographic data plane, geometric blue light shards, dynamic motion, ultra detailed\t97\t4.0\t3\t1\n"
        "cyberpunk warrior catching her breath against a neon wall, steam rising, cinematic close-up, atmospheric\t97\t4.0\t4\t1\n"
        "cyberpunk warrior walking away into the rain, back to camera, neon lights reflecting, cinematic ending shot\t97\t4.0\t5\t0\n"
    ),
    "[キャラ] ファンタジー冒険 5シーン": (
        "fantasy anime young hero with brown hair, green cloak and leather armor, standing at the edge of a forest looking toward distant mountains, golden hour, cinematic, ultra detailed\t97\t3.5\t0\t1\n"
        "same hero walking through a misty enchanted forest, glowing fireflies, soft fantasy lighting, cinematic\t97\t3.5\t1\t0\n"
        "hero in green cloak crossing an ancient stone bridge over a deep ravine, dramatic wide shot, fantasy art\t97\t3.5\t2\t0\n"
        "hero entering a ruined castle courtyard, sword drawn, sunlight piercing through broken walls, cinematic\t97\t3.5\t3\t0\n"
        "hero standing atop the castle ruins looking at a sunset sky with a dragon silhouette in the distance, epic fantasy\t97\t3.5\t4\t0\n"
    ),
    "[キャラ] ポートレート表情変化 4シーン": (
        "close-up portrait of a young woman with auburn hair and green eyes, soft natural light, calm neutral expression, cinematic, photorealistic\t73\t3.5\t0\t1\n"
        "same woman starting to smile softly, eyes brightening, gentle laugh, cinematic close-up, photorealistic\t73\t3.5\t1\t0\n"
        "same woman laughing genuinely, head tilting slightly back, joyful expression, cinematic, photorealistic\t73\t3.5\t2\t0\n"
        "same woman calming back to a peaceful smile, soft afternoon light, cinematic close-up, photorealistic\t73\t3.5\t3\t0\n"
    ),
    "[キャラ] 侍 抜刀シーン 5シーン": (
        "anime style samurai with black hair tied back wearing dark blue kimono and hakama, standing motionless in morning mist on a stone path, hand on sheathed katana, cinematic, dramatic\t97\t4.0\t0\t1\n"
        "same samurai slowly lowering his stance, focused eyes, hand tightening on the hilt, wind rustling the kimono, cinematic close-up\t73\t4.0\t1\t0\n"
        "samurai drawing the katana in a flash, motion blur trail of the blade, cherry petals in the air, dynamic action, cinematic\t73\t4.0\t2\t0\n"
        "single sweeping slash through the air, blade gleaming, dramatic side angle, cinematic action shot, anime style\t73\t4.0\t3\t0\n"
        "samurai sheathing the katana with finality, calm expression, mist returning around him, cinematic, atmospheric\t97\t4.0\t4\t0\n"
    ),
    "[キャラ] 巫女舞 4シーン": (
        "anime shrine maiden with long black hair wearing white kimono and red hakama, standing in a torii-lined shrine courtyard at dusk, lanterns glowing, cinematic, ultra detailed\t97\t3.5\t0\t1\n"
        "same shrine maiden beginning a slow ceremonial dance, holding bell-staff (kagura suzu), petals drifting, cinematic, anime style\t97\t3.5\t1\t0\n"
        "shrine maiden spinning gracefully, hakama flaring, soft blue light particles around her, cinematic peak moment, dynamic\t97\t3.5\t2\t0\n"
        "shrine maiden ending the dance with a low bow, lanterns swaying, peaceful silence, cinematic ending, anime style\t97\t3.5\t3\t0\n"
    ),
    "[キャラ] 魔法少女 変身バンク 5シーン": (
        "anime style ordinary schoolgirl with twin-tails in a school uniform raising a magical pendant, determined expression, magical sparkles starting around her, cinematic, cel-shaded\t73\t4.5\t0\t1\n"
        "intense burst of pink and white magical light enveloping her body, silhouette visible inside, dramatic, cinematic anime\t73\t4.5\t1\t0\n"
        "silhouette of the girl spinning inside the light vortex, ribbons of light forming, anime style transformation\t73\t4.5\t2\t0\n"
        "magical girl outfit forming layer by layer, frilly pink and white dress, ribbons, sparkles, cinematic, ultra detailed\t73\t4.5\t3\t0\n"
        "fully transformed magical girl landing on the ground with a pose, glowing wand, dramatic backlight, cinematic finale, cel-shaded\t97\t4.5\t4\t0\n"
    ),
    "[キャラ] カフェ店員 1日 5シーン": (
        "anime style female cafe staff with short brown hair wearing a brown apron, opening a cozy wooden cafe in the early morning, soft warm light, cinematic, ultra detailed\t97\t3.5\t0\t1\n"
        "same cafe staff carefully arranging pastries in the display case, gentle smile, cinematic close-up, anime style\t97\t3.5\t1\t0\n"
        "cafe staff pouring latte art into a cup, steam rising, focus shot, cinematic, anime style\t73\t3.5\t2\t0\n"
        "cafe staff serving a customer at a sunlit window seat, cheerful interaction, cinematic, anime style\t97\t3.5\t3\t0\n"
        "cafe staff wiping the counter at closing time, warm sunset light through the window, peaceful, cinematic, anime style\t97\t3.5\t4\t0\n"
    ),
    "[キャラ] ダンサー ステージ 5シーン": (
        "spotlight hitting an empty dance stage, stage smoke drifting, dramatic atmosphere, cinematic, photorealistic\t73\t3.5\t0\t1\n"
        "young female dancer with short black hair in a flowing white costume stepping into the spotlight, calm pose, cinematic\t97\t3.5\t1\t0\n"
        "same dancer flowing through a graceful spin, costume swirling, motion blur, cinematic, photorealistic\t97\t3.5\t2\t0\n"
        "dancer leaping into a high jump, freeze-frame feel, dramatic light from above, cinematic action\t73\t3.5\t3\t0\n"
        "dancer landing in a final pose, head tilted up, stage smoke around her, cinematic closing shot, photorealistic\t97\t3.5\t4\t0\n"
    ),
    "[キャラ] 探偵 推理シーン 4シーン": (
        "noir style young male detective with messy black hair in a long beige coat entering a dimly lit study, warm desk lamp, cinematic, photorealistic\t97\t3.5\t0\t1\n"
        "same detective kneeling to examine a clue on the wooden floor, magnifying glass in hand, dramatic shadow, cinematic close-up\t97\t3.5\t1\t0\n"
        "detective rising slowly with a thoughtful expression, fingers on his chin, dust particles in the lamp light, cinematic\t97\t3.5\t2\t0\n"
        "detective's eyes widening as he realizes the truth, dramatic backlight, cinematic noir, photorealistic\t73\t3.5\t3\t0\n"
    ),

    # ========================================================================
    # アニメ・ファンタジー演出
    # ========================================================================
    "[アニメ] 魔法エフェクト連発 5シーン": (
        "anime style scene of a glowing magic circle forming on a marble floor in a dark library, blue runes, particles, cinematic, ultra detailed\t97\t4.0\t0\t1\n"
        "magic circle erupting upward into a swirling pillar of blue energy, dramatic light, cinematic, anime style\t97\t4.0\t1\t0\n"
        "anime style young mage with white hair raising her staff, magical wind blowing her cloak, blue particles surrounding her, dynamic\t97\t4.0\t2\t1\n"
        "same mage casting a beam of light into the sky, dramatic energy explosion, anime style, cinematic\t97\t4.0\t3\t0\n"
        "calm aftermath, magic particles slowly drifting down in a quiet library, soft glow, anime style, atmospheric\t97\t4.0\t4\t0\n"
    ),
    "[アニメ] 必殺技発動 6シーン": (
        "anime style young male hero with spiky orange hair and torn battle outfit, taking a wide stance on a cracked battlefield, dramatic backlight, cinematic\t73\t4.5\t0\t1\n"
        "same hero clenching his fists, intense yellow aura igniting around him, ground cracking beneath his feet, cinematic, dynamic anime\t73\t4.5\t1\t0\n"
        "hero with golden energy aura swirling violently, hair rising, lightning crackling around him, dramatic, anime style\t73\t4.5\t2\t0\n"
        "hero raising both hands above his head to gather a glowing energy sphere, screen filling with light, cinematic, anime\t73\t4.5\t3\t0\n"
        "hero unleashing a massive beam of golden light forward, shockwave radiating, dramatic perspective, cinematic action\t97\t4.5\t4\t0\n"
        "wide shot of the battlefield after the blast, smoke and golden particles drifting, hero standing exhausted, cinematic finale\t97\t4.5\t5\t0\n"
    ),

    # ========================================================================
    # 映画風
    # ========================================================================
    "[映画風] ホラー 4シーン": (
        "horror movie shot of a dark abandoned hallway lit only by a flickering ceiling lamp, peeling wallpaper, cinematic, photorealistic\t97\t4.0\t0\t1\n"
        "slow push-in down the hallway toward a half-open door at the end, ominous atmosphere, cinematic, photorealistic\t97\t4.0\t1\t0\n"
        "the door slowly creaking open by itself, deep shadow inside, dust drifting, cinematic horror, photorealistic\t97\t4.0\t2\t0\n"
        "a ghostly pale face barely visible in the doorway shadow, deeply unsettling, cinematic horror jump moment, photorealistic\t73\t4.0\t3\t0\n"
    ),
    "[映画風] アクション カーチェイス 5シーン": (
        "wide cinematic shot of a black sports car speeding through a downtown street at night, neon reflections, motion blur, photorealistic\t73\t4.0\t0\t1\n"
        "low angle of the car drifting around a tight corner, sparks flying from the tires, dramatic, cinematic action\t73\t4.0\t1\t0\n"
        "interior driver POV through the windshield, dashboard glowing, oncoming traffic in slow motion, cinematic, photorealistic\t73\t4.0\t2\t0\n"
        "car jumping over a rising drawbridge gap, freeze-frame mid-air, dramatic side angle, cinematic action\t73\t4.0\t3\t0\n"
        "car landing hard on the far side with a shower of sparks, accelerating into the distance, cinematic finale shot\t97\t4.0\t4\t0\n"
    ),
    "[映画風] ロマンス 出会い 4シーン": (
        "rainy parisian street cafe terrace at evening, warm cafe lights, soft jazz mood, cinematic, photorealistic\t97\t3.5\t0\t1\n"
        "young woman in a red coat looking up from her book as a stranger sits at the next table, eyes meeting briefly, cinematic close-up\t97\t3.5\t1\t0\n"
        "same young man smiling gently and offering an umbrella, both bathed in warm cafe light, cinematic, photorealistic\t97\t3.5\t2\t0\n"
        "two of them walking together away from the cafe under the shared umbrella, soft rain, cinematic ending shot, photorealistic\t97\t3.5\t3\t0\n"
    ),

    # ========================================================================
    # 抽象・エフェクト（被写体に縛られない映像)
    # ========================================================================
    "[抽象] 流体カラフル 4シーン": (
        "abstract macro shot of colorful ink swirling in clear water, deep blue and magenta clouds blooming, slow motion, cinematic, ultra detailed\t97\t3.0\t0\t1\n"
        "new orange ink dropping into the same water, expanding into intricate fractal-like tendrils, slow motion, cinematic\t97\t3.0\t1\t0\n"
        "ink colors mixing into a swirling vortex, dramatic light from above, ultra detailed abstract\t97\t3.0\t2\t0\n"
        "slow settling of the inks into a calm, layered pattern, soft light, cinematic abstract, ultra detailed\t97\t3.0\t3\t0\n"
    ),
    "[抽象] 粒子の踊り 4シーン": (
        "abstract dark void with tiny golden particles slowly appearing and floating freely, cinematic, ultra detailed\t97\t3.0\t0\t1\n"
        "golden particles drifting together to form a swirling sphere of light, dramatic, cinematic abstract\t97\t3.0\t1\t0\n"
        "particle sphere collapsing into a vertical column of light, dramatic upward motion, cinematic abstract\t97\t3.0\t2\t0\n"
        "column of light dispersing back into thousands of golden particles drifting outward, peaceful, cinematic\t97\t3.0\t3\t0\n"
    ),
}
