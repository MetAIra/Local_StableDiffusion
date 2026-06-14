"""Long Prompt エンコーディングユーティリティ

CLIPの77トークン制限を超えるプロンプトを複数チャンクに分割して
エンコードし、結合することで全トークンを画像生成に反映する。

SD1.5: text_encoder(input_ids)[0] = last_hidden_state を使用
SDXL:  text_encoder(input_ids, output_hidden_states=True).hidden_states[-2] を使用
"""
import torch
from typing import Optional


def encode_prompt_long(
    pipe,
    prompt: str,
    negative_prompt: str = "",
    device: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """77トークン超のプロンプトをチャンク分割でエンコード（SD1.5用）

    Args:
        pipe: StableDiffusionPipeline（tokenizer, text_encoder を持つ）
        prompt: ポジティブプロンプト
        negative_prompt: ネガティブプロンプト
        device: デバイス（Noneならtext_encoderのデバイスを使用）

    Returns:
        (prompt_embeds, negative_prompt_embeds)
    """
    if device is None:
        device = pipe.text_encoder.device

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # SD1.5: last_hidden_state を使用
    prompt_embeds = _encode_single_prompt(tokenizer, text_encoder, prompt, device, use_penultimate=False)
    neg_embeds = _encode_single_prompt(tokenizer, text_encoder, negative_prompt or "", device, use_penultimate=False)

    # positiveとnegativeのシーケンス長を揃える（UNetのcross-attentionで必要）
    max_len = max(prompt_embeds.shape[1], neg_embeds.shape[1])
    if prompt_embeds.shape[1] < max_len:
        prompt_embeds = _pad_embeds(prompt_embeds, max_len, tokenizer, text_encoder, device, use_penultimate=False)
    if neg_embeds.shape[1] < max_len:
        neg_embeds = _pad_embeds(neg_embeds, max_len, tokenizer, text_encoder, device, use_penultimate=False)

    return prompt_embeds, neg_embeds


def encode_prompt_long_sdxl(
    pipe,
    prompt: str,
    negative_prompt: str = "",
    device: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """77トークン超のプロンプトをチャンク分割でエンコード（SDXL用）

    Args:
        pipe: StableDiffusionXLPipeline
        prompt: ポジティブプロンプト
        negative_prompt: ネガティブプロンプト
        device: デバイス

    Returns:
        (prompt_embeds, negative_prompt_embeds,
         pooled_prompt_embeds, negative_pooled_prompt_embeds)
    """
    if device is None:
        device = pipe.text_encoder.device

    # SDXL: hidden_states[-2] を使用（両エンコーダとも）
    # Encoder 1
    embeds_1 = _encode_single_prompt(
        pipe.tokenizer, pipe.text_encoder, prompt, device, use_penultimate=True
    )
    neg_embeds_1 = _encode_single_prompt(
        pipe.tokenizer, pipe.text_encoder, negative_prompt or "", device, use_penultimate=True
    )

    # Encoder 2 (with pooled output)
    embeds_2, pooled = _encode_single_prompt_with_pooled(
        pipe.tokenizer_2, pipe.text_encoder_2, prompt, device
    )
    neg_embeds_2, neg_pooled = _encode_single_prompt_with_pooled(
        pipe.tokenizer_2, pipe.text_encoder_2, negative_prompt or "", device
    )

    # シーケンス長を揃える
    max_len = max(embeds_1.shape[1], neg_embeds_1.shape[1],
                  embeds_2.shape[1], neg_embeds_2.shape[1])

    if embeds_1.shape[1] < max_len:
        embeds_1 = _pad_embeds(embeds_1, max_len, pipe.tokenizer, pipe.text_encoder, device, use_penultimate=True)
    if neg_embeds_1.shape[1] < max_len:
        neg_embeds_1 = _pad_embeds(neg_embeds_1, max_len, pipe.tokenizer, pipe.text_encoder, device, use_penultimate=True)
    if embeds_2.shape[1] < max_len:
        embeds_2 = _pad_embeds(embeds_2, max_len, pipe.tokenizer_2, pipe.text_encoder_2, device, use_penultimate=True)
    if neg_embeds_2.shape[1] < max_len:
        neg_embeds_2 = _pad_embeds(neg_embeds_2, max_len, pipe.tokenizer_2, pipe.text_encoder_2, device, use_penultimate=True)

    # 2つのエンコーダの出力を結合（hidden_dim方向）
    prompt_embeds = torch.cat([embeds_1, embeds_2], dim=-1)
    neg_prompt_embeds = torch.cat([neg_embeds_1, neg_embeds_2], dim=-1)

    return prompt_embeds, neg_prompt_embeds, pooled, neg_pooled


def get_token_count(pipe, prompt: str) -> int:
    """プロンプトのトークン数を返す"""
    tokens = pipe.tokenizer(
        prompt,
        truncation=False,
        add_special_tokens=False,
    ).input_ids
    return len(tokens)


def needs_long_encoding(pipe, prompt: str, negative_prompt: str = "") -> bool:
    """プロンプトが77トークンを超えるか判定"""
    max_content = pipe.tokenizer.model_max_length - 2  # BOS + EOS分を除く
    pos_count = get_token_count(pipe, prompt)
    neg_count = get_token_count(pipe, negative_prompt) if negative_prompt else 0
    return pos_count > max_content or neg_count > max_content


def encode_long_prompts_if_needed(
    pipe,
    full_prompt: str,
    full_negative: str,
    is_sdxl: bool,
) -> tuple[bool, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """77トークン超なら事前エンコードして埋め込みを返す

    Returns:
        (use_long_prompt, prompt_embeds, negative_prompt_embeds,
         pooled_prompt_embeds, negative_pooled_prompt_embeds)
        短いプロンプトの場合は (False, None, None, None, None)。
    """
    if not needs_long_encoding(pipe, full_prompt, full_negative):
        return False, None, None, None, None

    from config import DEVICE

    token_count = get_token_count(pipe, full_prompt)
    print(f"Long Prompt detected: {token_count} tokens (limit: 75). Using chunked encoding.")
    if is_sdxl:
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
            encode_prompt_long_sdxl(pipe, full_prompt, full_negative, DEVICE)
        )
        return True, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    prompt_embeds, negative_prompt_embeds = (
        encode_prompt_long(pipe, full_prompt, full_negative, DEVICE)
    )
    return True, prompt_embeds, negative_prompt_embeds, None, None


def build_long_prompt_kwargs(
    is_sdxl: bool,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    **extra,
) -> dict:
    """Long Prompt用のパイプライン呼び出しkwargsを組み立てる

    extra には image / mask_image / width / height / strength /
    num_inference_steps / guidance_scale / generator など
    パイプライン固有の引数を渡す。
    """
    gen_kwargs = dict(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        **extra,
    )
    if is_sdxl and pooled_prompt_embeds is not None:
        gen_kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds
        gen_kwargs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
    return gen_kwargs


# --- internal helpers ---

def _encode_single_prompt(
    tokenizer,
    text_encoder,
    prompt: str,
    device,
    use_penultimate: bool = False,
) -> torch.Tensor:
    """単一のプロンプトをチャンク分割でエンコード

    Args:
        use_penultimate:
            False → output[0] (last_hidden_state) を使用（SD1.5標準）
            True  → hidden_states[-2] を使用（SDXL標準）
    """
    max_length = tokenizer.model_max_length  # 77
    chunk_content_size = max_length - 2  # BOS + EOS を除く: 75

    # truncation=False でフルトークン列を取得
    tokens = tokenizer(
        prompt,
        truncation=False,
        add_special_tokens=False,
    ).input_ids

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos

    # チャンクに分割
    chunks = []
    for i in range(0, max(len(tokens), 1), chunk_content_size):
        content = tokens[i:i + chunk_content_size]
        # [BOS] + content + [EOS] + padding
        padded = [bos] + content + [eos]
        padded += [pad] * (max_length - len(padded))
        chunks.append(padded[:max_length])

    if not chunks:
        chunks = [[bos, eos] + [pad] * (max_length - 2)]

    # 各チャンクをエンコードして結合
    all_embeds = []
    for chunk in chunks:
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        with torch.no_grad():
            if use_penultimate:
                # SDXL: penultimate hidden state (hidden_states[-2])
                output = text_encoder(input_ids, output_hidden_states=True)
                hidden = output.hidden_states[-2]
            else:
                # SD1.5: last_hidden_state (output[0])
                output = text_encoder(input_ids)
                hidden = output[0]
        all_embeds.append(hidden)

    # シーケンス方向に結合: (1, 77*N, hidden_dim)
    return torch.cat(all_embeds, dim=1)


def _encode_single_prompt_with_pooled(
    tokenizer,
    text_encoder,
    prompt: str,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """単一のプロンプトをエンコード（pooled outputも返す、SDXL encoder2用）

    SDXL encoder2 は hidden_states[-2] を使用。
    pooled output は最初のチャンクの output[0] から取得。
    """
    max_length = tokenizer.model_max_length
    chunk_content_size = max_length - 2

    tokens = tokenizer(
        prompt,
        truncation=False,
        add_special_tokens=False,
    ).input_ids

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos

    chunks = []
    for i in range(0, max(len(tokens), 1), chunk_content_size):
        content = tokens[i:i + chunk_content_size]
        padded = [bos] + content + [eos]
        padded += [pad] * (max_length - len(padded))
        chunks.append(padded[:max_length])

    if not chunks:
        chunks = [[bos, eos] + [pad] * (max_length - 2)]

    all_embeds = []
    pooled_output = None

    for idx, chunk in enumerate(chunks):
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        with torch.no_grad():
            output = text_encoder(input_ids, output_hidden_states=True)
            hidden = output.hidden_states[-2]
            # pooled outputは最初のチャンクから取得（全体の要約）
            if idx == 0:
                pooled_output = output[0]
        all_embeds.append(hidden)

    embeds = torch.cat(all_embeds, dim=1)
    return embeds, pooled_output


def _pad_embeds(
    embeds: torch.Tensor,
    target_len: int,
    tokenizer,
    text_encoder,
    device,
    use_penultimate: bool = False,
) -> torch.Tensor:
    """埋め込みをtarget_lenまでパディング（空トークンのエンコード結果で埋める）"""
    current_len = embeds.shape[1]
    if current_len >= target_len:
        return embeds

    max_length = tokenizer.model_max_length
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    # パディング用の空チャンクを必要なだけ生成
    pad_needed = target_len - current_len
    pad_chunks_needed = (pad_needed + max_length - 1) // max_length

    pad_embeds_list = []
    for _ in range(pad_chunks_needed):
        pad_input = [bos, eos] + [pad] * (max_length - 2)
        input_ids = torch.tensor([pad_input], dtype=torch.long, device=device)
        with torch.no_grad():
            if use_penultimate:
                output = text_encoder(input_ids, output_hidden_states=True)
                hidden = output.hidden_states[-2]
            else:
                output = text_encoder(input_ids)
                hidden = output[0]
        pad_embeds_list.append(hidden)

    pad_embeds = torch.cat(pad_embeds_list, dim=1)
    # 必要な分だけトリム
    pad_embeds = pad_embeds[:, :pad_needed, :]

    return torch.cat([embeds, pad_embeds], dim=1)
