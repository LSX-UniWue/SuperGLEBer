# fix error for leo hessian

import torch
import flair
from typing import Optional
from flair.embeddings import TransformerEmbeddings


@torch.jit.script_if_tracing
def truncate_hidden_states(hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    return hidden_states[:, :, : input_ids.size(1)]


@torch.jit.script_if_tracing
def combine_strided_tensors(
        hidden_states: torch.Tensor,
        overflow_to_sample_mapping: torch.Tensor,
        half_stride: int,
        max_length: int,
        default_value: int,
) -> torch.Tensor:
    _, counts = torch.unique(overflow_to_sample_mapping, sorted=True, return_counts=True)
    sentence_count = int(overflow_to_sample_mapping.max().item() + 1)
    token_count = max_length + (max_length - 2) * int(counts.max().item() - 1)
    if hidden_states.dim() == 2:
        sentence_hidden_states = torch.zeros(
            (sentence_count, token_count), device=flair.device, dtype=hidden_states.dtype
        )
    else:
        sentence_hidden_states = torch.zeros(
            (sentence_count, token_count, hidden_states.shape[2]), device=flair.device, dtype=hidden_states.dtype
        )
    sentence_hidden_states += default_value
    for sentence_id in torch.arange(0, sentence_hidden_states.shape[0]):
        selected_sentences = hidden_states[overflow_to_sample_mapping == sentence_id]
        if selected_sentences.size(0) > 1:
            start_part = selected_sentences[0, : half_stride + 1]
            mid_part = selected_sentences[:, half_stride + 1: max_length - 1 - half_stride]
            mid_part = torch.reshape(mid_part, (mid_part.size(0) * mid_part.size(1),) + mid_part.size()[2:])
            end_part = selected_sentences[selected_sentences.size(0) - 1, max_length - half_stride - 1:]
            sentence_hidden_state = torch.cat((start_part, mid_part, end_part), dim=0)
            sentence_hidden_states[sentence_id, : sentence_hidden_state.size(0)] = sentence_hidden_state
        else:
            sentence_hidden_states[sentence_id, : selected_sentences.size(1)] = selected_sentences[0, :]
    return sentence_hidden_states


@torch.jit.script_if_tracing
def insert_missing_embeddings(
        token_embeddings: torch.Tensor, word_id: torch.Tensor, length: torch.LongTensor
) -> torch.Tensor:
    if token_embeddings.shape[0] == 0:
        if token_embeddings.dim() == 2:
            token_embeddings = torch.zeros(
                int(length), token_embeddings.shape[1], dtype=token_embeddings.dtype, device=token_embeddings.device
            )
    elif token_embeddings.shape[0] < length:
        for _id in torch.arange(int(length)):
            zero_vector = torch.zeros_like(token_embeddings[:1])
            if not (word_id == _id).any():
                token_embeddings = torch.cat(
                    (
                        token_embeddings[:_id],
                        zero_vector,
                        token_embeddings[_id:],
                    ),
                    dim=0,
                )
    return token_embeddings


@torch.jit.script_if_tracing
def fill_masked_elements(
        all_token_embeddings: torch.Tensor,
        sentence_hidden_states: torch.Tensor,
        mask: torch.Tensor,
        word_ids: torch.Tensor,
        lengths: torch.LongTensor,
):
    for i in torch.arange(int(all_token_embeddings.shape[0])):
        r = insert_missing_embeddings(sentence_hidden_states[i][mask[i] & (word_ids[i] >= 0)], word_ids[i], lengths[i])
        all_token_embeddings[i, : lengths[i], :] = r
    return all_token_embeddings


@torch.jit.script_if_tracing
def fill_mean_token_embeddings(
        all_token_embeddings: torch.Tensor,
        sentence_hidden_states: torch.Tensor,
        word_ids: torch.Tensor,
        token_lengths: torch.Tensor,
):
    batch_size, max_tokens, embedding_dim = all_token_embeddings.shape
    mask = word_ids >= 0
    all_token_embeddings.scatter_add_(
        1,
        word_ids.clamp(min=0).unsqueeze(-1).expand(-1, -1, embedding_dim),
        sentence_hidden_states * mask.unsqueeze(-1).to(all_token_embeddings.dtype),
    )
    subtoken_counts = torch.zeros_like(all_token_embeddings[:, :, 0])
    subtoken_counts.scatter_add_(1, word_ids.clamp(min=0), mask.to(subtoken_counts.dtype))
    all_token_embeddings = torch.where(
        subtoken_counts.unsqueeze(-1) > 0,
        all_token_embeddings / subtoken_counts.unsqueeze(-1),
        torch.zeros_like(all_token_embeddings),
    )
    token_mask = torch.arange(max_tokens, device=token_lengths.device)[None, :] < token_lengths[:, None]
    all_token_embeddings = all_token_embeddings * token_mask.unsqueeze(-1)
    all_token_embeddings = torch.nan_to_num(all_token_embeddings)
    return all_token_embeddings


@torch.jit.script_if_tracing
def document_cls_pooling(sentence_hidden_states: torch.Tensor, sentence_lengths: torch.Tensor) -> torch.Tensor:
    return sentence_hidden_states[torch.arange(sentence_hidden_states.shape[0]), sentence_lengths - 1]


@torch.jit.script_if_tracing
def document_mean_pooling(sentence_hidden_states: torch.Tensor, sentence_lengths: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(
        sentence_hidden_states.shape[0], sentence_hidden_states.shape[2], dtype=sentence_hidden_states.dtype,
        device=flair.device
    )
    for i in torch.arange(sentence_hidden_states.shape[0]):
        result[i] = sentence_hidden_states[i, : sentence_lengths[i]].mean(dim=0)
    return result


@torch.jit.script_if_tracing
def document_max_pooling(sentence_hidden_states: torch.Tensor, sentence_lengths: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(
        sentence_hidden_states.shape[0], sentence_hidden_states.shape[2], dtype=sentence_hidden_states.dtype,
        device=flair.device
    )
    for i in torch.arange(sentence_hidden_states.shape[0]):
        result[i], _ = sentence_hidden_states[i, : sentence_lengths[i]].max(dim=0)
    return result


# --- The Patched Forward Method ---
def patched_transformer_forward(
        self,
        input_ids: torch.Tensor,
        sub_token_lengths: Optional[torch.LongTensor] = None,
        token_lengths: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        overflow_to_sample_mapping: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
):
    model_kwargs = {}
    if langs is not None:
        model_kwargs["langs"] = langs
    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask
    if bbox is not None:
        model_kwargs["bbox"] = bbox
    if pixel_values is not None:
        model_kwargs["pixel_values"] = pixel_values

    # fixie
    model_output = self.model(input_ids, output_hidden_states=True, **model_kwargs)
    hidden_states = model_output.hidden_states

    hidden_states = torch.stack(hidden_states)

    hidden_states = truncate_hidden_states(hidden_states, input_ids)

    hidden_states = hidden_states[self.layer_indexes, :, :]
    if self.layer_mean:
        hidden_states = hidden_states.mean(dim=0)
    else:
        hidden_states = torch.flatten(hidden_states.permute((0, 3, 1, 2)), 0, 1).permute((1, 2, 0))
    if self._can_document_embedding_shortcut():
        return {"document_embeddings": hidden_states[:, 0]}
    if self.allow_long_sentences:
        assert overflow_to_sample_mapping is not None
        sentence_hidden_states = combine_strided_tensors(
            hidden_states, overflow_to_sample_mapping, self.stride // 2, self.tokenizer.model_max_length, 0
        )
        if self.tokenizer.is_fast and self.token_embedding:
            word_ids = combine_strided_tensors(
                word_ids, overflow_to_sample_mapping, self.stride // 2, self.tokenizer.model_max_length, -100
            )
    else:
        sentence_hidden_states = hidden_states
    result = {}
    if self.document_embedding:
        if self.cls_pooling == "cls" and self.initial_cls_token:
            document_embeddings = sentence_hidden_states[:, 0]
        else:
            assert sub_token_lengths is not None
            if self.cls_pooling == "cls":
                document_embeddings = document_cls_pooling(sentence_hidden_states, sub_token_lengths)
            elif self.cls_pooling == "mean":
                document_embeddings = document_mean_pooling(sentence_hidden_states, sub_token_lengths)
            elif self.cls_pooling == "max":
                document_embeddings = document_max_pooling(sentence_hidden_states, sub_token_lengths)
            else:
                raise ValueError(f"cls pooling method: `{self.cls_pooling}` is not implemented")
        result["document_embeddings"] = document_embeddings
    if self.token_embedding:
        assert word_ids is not None
        assert token_lengths is not None
        all_token_embeddings = torch.zeros(
            word_ids.shape[0],
            token_lengths.max(),
            self.embedding_length_internal,
            device=flair.device,
            dtype=sentence_hidden_states.dtype,
        )
        true_tensor = torch.ones_like(word_ids[:, :1], dtype=torch.bool)
        if self.subtoken_pooling == "first":
            gain_mask = word_ids[:, 1:] != word_ids[:, : word_ids.shape[1] - 1]
            first_mask = torch.cat([true_tensor, gain_mask], dim=1)
            all_token_embeddings = fill_masked_elements(
                all_token_embeddings, sentence_hidden_states, first_mask, word_ids, token_lengths
            )
        elif self.subtoken_pooling == "last":
            gain_mask = word_ids[:, 1:] != word_ids[:, : word_ids.shape[1] - 1]
            last_mask = torch.cat([gain_mask, true_tensor], dim=1)
            all_token_embeddings = fill_masked_elements(
                all_token_embeddings, sentence_hidden_states, last_mask, word_ids, token_lengths
            )
        elif self.subtoken_pooling == "first_last":
            gain_mask = word_ids[:, 1:] != word_ids[:, : word_ids.shape[1] - 1]
            first_mask = torch.cat([true_tensor, gain_mask], dim=1)
            last_mask = torch.cat([gain_mask, true_tensor], dim=1)
            all_token_embeddings[:, :, : sentence_hidden_states.shape[2]] = fill_masked_elements(
                all_token_embeddings[:, :, : sentence_hidden_states.shape[2]],
                sentence_hidden_states,
                first_mask,
                word_ids,
                token_lengths,
            )
            all_token_embeddings[:, :, sentence_hidden_states.shape[2]:] = fill_masked_elements(
                all_token_embeddings[:, :, sentence_hidden_states.shape[2]:],
                sentence_hidden_states,
                last_mask,
                word_ids,
                token_lengths,
            )
        elif self.subtoken_pooling == "mean":
            all_token_embeddings = fill_mean_token_embeddings(
                all_token_embeddings, sentence_hidden_states, word_ids, token_lengths
            )
        else:
            raise ValueError(f"subtoken pooling method: `{self.subtoken_pooling}` is not implemented")
        result["token_embeddings"] = all_token_embeddings
    return result


TransformerEmbeddings.forward = patched_transformer_forward
