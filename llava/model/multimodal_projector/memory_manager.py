import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int, 
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).
    """
    protected = 0

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    assert r > 0, r

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src) # , reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


class MemoryManager(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, st_memory_windows=[1, 18], st_memory_tokens=[729, 128], event_split_window=8,
            long_memory_tokens_per_frame=64, long_memory_tokens_quota=5120, sim_weight_g=0.4, time_weight_a=0.2, merge_weight_b=0.4,
            anomaly_pool_max_size=8, anomaly_pool_tokens=128, anomaly_pool_protect_recent=2,
            aps_lambda=0.05, aps_kappa=4.0, aps_weight=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        
        self.st_memory_windows = st_memory_windows
        self.st_memory_tokens = st_memory_tokens
        self.event_split_window = event_split_window
        self.long_memory_tokens_per_frame = long_memory_tokens_per_frame
        self.long_memory_tokens_quota = long_memory_tokens_quota
        self.long_memory_current_tokens = 0
        self.long_memory_last_time_pos = -1
        
        self.sim_weight_g = sim_weight_g
        self.time_weight_a = time_weight_a
        self.merge_weight_b = merge_weight_b

        # APS (Anomaly Priority Score) parameters for PEMF protection
        self.aps_lambda = aps_lambda    # base scale factor
        self.aps_kappa = aps_kappa      # exponential growth rate
        self.aps_weight = aps_weight    # weight in total_penalties formula

        # Anomaly Pool parameters
        self.anomaly_pool_max_size = anomaly_pool_max_size
        self.anomaly_pool_tokens = anomaly_pool_tokens
        self.anomaly_pool_protect_recent = anomaly_pool_protect_recent

        # memory
        self.memory_now = []
        self.memory_short = []
        self.memory_long = []
        
        # Anomaly Pool: visual token storage + score tracking
        self.anomaly_pool = []          # list of [1, anomaly_pool_tokens, C] tensors
        self.anomaly_pool_scores = []   # list of float scores (aligned with anomaly_pool)
        
        # long memory buffer
        self.time_buffer = []
        self.mergecnt_buffer = []
        self.similarity_buffer = []
        self.anomaly_score_buffer = []  # APS: per-event anomaly score for PEMF protection
        
        # short memory buffer: per-frame anomaly scores for event splitting
        self.last_frame_flat = None
        self.frame_sim_buffer = []
        self.now_anomaly_scores = []    # APS: anomaly scores aligned with memory_now
        self.short_anomaly_scores = []  # APS: anomaly scores aligned with memory_short

        self.reset()

    def reset(self):
        self.memory_now = []
        self.memory_short = []
        self.memory_long = []
        
        # Anomaly Pool
        self.anomaly_pool = []
        self.anomaly_pool_scores = []
        
        self.time_buffer = []
        self.mergecnt_buffer = []
        self.similarity_buffer = []
        self.anomaly_score_buffer = []
        
        self.last_frame_flat = None
        self.frame_sim_buffer = []
        self.now_anomaly_scores = []
        self.short_anomaly_scores = []
        
        self.long_memory_current_tokens = 0
        self.long_memory_last_time_pos = -1
        
        torch.cuda.empty_cache()
        
    def merge_tokens(self, x, target_num_token):
        r"""
        x = torch.randn(10, 2560, c)
        x = merge_tokens(x, r_merge_list=[1280])
        """
        size = None
        b, p, c = x.shape
        tmp_p = p
        r_merge_list = []
        
        if tmp_p == target_num_token:  #not compress
            return x
        
        assert tmp_p > target_num_token, f"{tmp_p} should greater than {target_num_token}"
        while tmp_p != target_num_token:
            if tmp_p - target_num_token <= (tmp_p // 2):
                r_merge_list.append(tmp_p - target_num_token)
                break
            else:
                r_merge_list.append(tmp_p // 2)
                tmp_p = tmp_p - (tmp_p // 2)
                
        
        head = self.num_attention_heads

        dim = c // head
        for r in r_merge_list:
            metric = x.reshape(b, p, head, dim).mean(2) # [b, p, c//head]
            merge, _ = bipartite_soft_matching(
                metric, 
                r
            )
            x, size = merge_wavg(merge, x, size)
            _, p, _ = x.shape
        # x = x.reshape(-1, c)  # 300, 1024
        return x

    def calculate_similarity_between_clips(self, clip1, clip2):
        new_clip = torch.cat([clip1, clip2], dim=0).reshape(-1, clip1.shape[-1])  # [T, C]
        total_tokens = new_clip.shape[0]
        target_tokens = math.ceil((clip1.shape[0] + clip2.shape[0]) / 2) * self.long_memory_tokens_per_frame

        p, c = new_clip.shape
        head = self.num_attention_heads
        dim = c // head
        r = total_tokens - target_tokens  
        assert r <= total_tokens // 2, "The merged tokens must not exceed half! "
        assert r > 0, "Token merging count r must be > 0! "
        
        with torch.no_grad():
            new_clip = new_clip.reshape(p, head, dim).mean(1)
            
            x = new_clip / new_clip.norm(dim=-1, keepdim=True)  
            a, b = x[::2], x[1::2]  
            scores = a @ b.transpose(-1, -2)  
            
            max_scores, _ = scores.max(dim=-1)  
            top_r_scores = max_scores.topk(r, largest=True).values  

        return top_r_scores.mean().item() 

    def _update_short_memory(self):
        overflow = len(self.memory_now) - self.st_memory_windows[0]
        if overflow > 0:
            old_now_batch = self.memory_now[:overflow]
            self.memory_now = self.memory_now[overflow:]
            
            # APS: move anomaly scores from Now to Short (aligned with memory_short)
            self.short_anomaly_scores.extend(self.now_anomaly_scores[:overflow])
            self.now_anomaly_scores = self.now_anomaly_scores[overflow:]
            
            old_now_batch = torch.cat(old_now_batch, dim=0)  # [B, p, c]
            short_tokens_batch = self.merge_tokens(old_now_batch, target_num_token=self.st_memory_tokens[1])  # [B, p', c]
            self.memory_short.extend(t.unsqueeze(0) for t in short_tokens_batch.unbind(0))
            
            del old_now_batch, short_tokens_batch
        return


        
    def _update_event_split(self):
        while len(self.memory_short) >= self.st_memory_windows[1] + self.event_split_window:
            # print("<<<Mark>>> self.memory_short: ", len(self.memory_short), "self.frame_sim_buffer: ", len(self.frame_sim_buffer))
            
            window_sim = torch.stack(self.frame_sim_buffer[:self.event_split_window])
            # print("<<<Mark>>> window_sim len:", len(window_sim), "window_sim:", window_sim)
            
            min_sim_idx = torch.argmin(window_sim)
            # print("<<<Mark>>> min_sim_idx:", min_sim_idx)
            
            split_frame_idx = min_sim_idx + 1
            # print("<<<Mark>>> split_frame_idx:", split_frame_idx)
            
            old_short = torch.cat(self.memory_short[:split_frame_idx], dim=0)
            # print("<<<Mark>>> old_short:", old_short.shape)
            
            # APS: compute event-level anomaly score from per-frame scores
            event_frame_scores = self.short_anomaly_scores[:split_frame_idx]
            event_anomaly_score = max(event_frame_scores) if event_frame_scores else 0.0
            
            self.memory_short = self.memory_short[split_frame_idx:]     # delete old short memory
            self.frame_sim_buffer = self.frame_sim_buffer[split_frame_idx:]
            self.short_anomaly_scores = self.short_anomaly_scores[split_frame_idx:]
            
            # print("<<<Mark>>> self.memory_short: ", len(self.memory_short), "self.frame_sim_buffer: ", len(self.frame_sim_buffer))
            
            event_merged = self.merge_tokens(old_short.reshape(1, -1, self.hidden_size), self.long_memory_tokens_per_frame*old_short.shape[0]).reshape(old_short.shape[0], -1, self.hidden_size)
            # print("<<<Mark>>> event_merged ", event_merged.shape)
            
            # Initialize long memory
            self.memory_long.append(event_merged)
            self.long_memory_current_tokens += event_merged.shape[0]*event_merged.shape[1]
            self.time_buffer.append((self.long_memory_last_time_pos*2 + split_frame_idx + 1)/2)
            self.long_memory_last_time_pos+=split_frame_idx
            self.mergecnt_buffer.append(1)
            self.anomaly_score_buffer.append(event_anomaly_score)  # APS: track per-event anomaly score
            if len(self.memory_long)>1:
                self.similarity_buffer.append(self.calculate_similarity_between_clips(self.memory_long[-2], self.memory_long[-1]))

            del old_short, event_merged, window_sim
        return

    def _update_long_memory(self):
        while self.long_memory_current_tokens > self.long_memory_tokens_quota and len(self.memory_long) > 1:
            input_device = self.memory_long[0].device
            num_events = len(self.memory_long)

            # 1. calculate overall pan
            sim_scores = torch.tensor(self.similarity_buffer, device=input_device)  # [num_events-1]
            time_diffs = torch.tensor([(self.time_buffer[i+1] + self.time_buffer[i])/2 for i in range(num_events-1)], device = input_device)
            merge_cnts = torch.tensor([(self.mergecnt_buffer[i] + self.mergecnt_buffer[i+1])/2 for i in range(num_events-1)], device = input_device)

            sim_penalties = 1 - sim_scores
            time_penalties = time_diffs / (self.long_memory_last_time_pos + len(self.memory_short) + len(self.memory_now))
            merge_penalties = merge_cnts / merge_cnts.max() + 1e-6

            total_penalties = self.sim_weight_g * sim_penalties    +    self.time_weight_a * time_penalties    +    self.merge_weight_b * merge_penalties

            # APS (Anomaly Priority Score): protect anomaly-containing events from merging.
            # APS = lambda * exp(kappa * S), where S is the max anomaly score of
            # an adjacent event pair. Because the merge candidate is selected with
            # argmin(total_penalties), higher APS is added to the penalty so
            # high-score anomaly pairs are less likely to be merged.
            #
            # Normalization strategy: Fixed upper bound (NOT dynamic max)
            # - Dynamic max normalization (aps / aps.max()) causes problems:
            #   In normal videos where all scores are low (e.g., 0.1~0.3), the max is also low,
            #   causing normalization to amplify these low scores to 40%~100% protection.
            # - Fixed upper bound: aps_max = λ · e^(κ · 1.0) (assuming max possible score = 1.0)
            #   This ensures consistent APS values across different videos:
            #   score=0.0 → ~2% protection, score=0.4 → ~9%, score=1.0 → 100%
            if self.aps_weight > 0 and len(self.anomaly_score_buffer) == num_events:
                pair_anomaly_scores = torch.tensor(
                    [max(self.anomaly_score_buffer[i], self.anomaly_score_buffer[i+1]) 
                     for i in range(num_events - 1)],
                    device=input_device
                )
                aps_values = self.aps_lambda * torch.exp(self.aps_kappa * pair_anomaly_scores)
                # Fixed upper bound normalization: use max possible score (1.0) as reference
                # This ensures normal events (~2% protection) vs anomaly events (~100% protection)
                aps_max = self.aps_lambda * math.exp(self.aps_kappa * 1.0)
                aps_values = aps_values / (aps_max + 1e-8)  # [0, 1] range with fixed reference
                total_penalties = total_penalties + self.aps_weight * aps_values

            merge_idx = torch.argmin(total_penalties).item()
            # print("<_update_long_memory> merge_idx: ", merge_idx)
            # 2. conduct merge
            clip1, clip2 = self.memory_long[merge_idx], self.memory_long[merge_idx+1]
            # print("<_update_long_memory> clip1: ", clip1.shape, "clip2", clip2.shape)
            merged_tokens = torch.cat([clip1, clip2], dim=0).reshape(1, -1, self.hidden_size)  # [1, T, C]
            # print("<_update_long_memory> merged_tokens before: ", merged_tokens.shape)
            merged_tokens = self.merge_tokens(merged_tokens, self.long_memory_tokens_per_frame * math.ceil((clip1.shape[0] + clip2.shape[0])/2)).reshape(-1, self.long_memory_tokens_per_frame, self.hidden_size)  # [T', C]
            # print("<_update_long_memory> merged_tokens after: ", merged_tokens.shape)
            
            
            self.memory_long[merge_idx] = merged_tokens
            del self.memory_long[merge_idx+1]
            
            self.long_memory_current_tokens -= (clip1.shape[0] + clip2.shape[0] - merged_tokens.shape[0]) * self.long_memory_tokens_per_frame

            # 3. update time_buffer
            self.time_buffer[merge_idx] = (self.time_buffer[merge_idx] * clip1.shape[0] + self.time_buffer[merge_idx+1] * clip2.shape[0]) / (clip1.shape[0] + clip2.shape[0])
            del self.time_buffer[merge_idx+1]

            # 4. update mergecnt_buffer
            self.mergecnt_buffer[merge_idx] += self.mergecnt_buffer[merge_idx+1]
            del self.mergecnt_buffer[merge_idx+1]

            # 4.5 APS: update anomaly_score_buffer (merged event inherits max score)
            if len(self.anomaly_score_buffer) > merge_idx + 1:
                self.anomaly_score_buffer[merge_idx] = max(
                    self.anomaly_score_buffer[merge_idx],
                    self.anomaly_score_buffer[merge_idx+1]
                )
                del self.anomaly_score_buffer[merge_idx+1]

            # 5. update similarity_buffer locally
            del self.similarity_buffer[merge_idx]
            # update left pair (merge_idx-1, merge_idx)
            if merge_idx - 1 >= 0:
                self.similarity_buffer[merge_idx-1] = self.calculate_similarity_between_clips(self.memory_long[merge_idx-1], self.memory_long[merge_idx])
            # update right pair (merge_idx, merge_idx+1)
            if merge_idx < len(self.memory_long) - 1:
                self.similarity_buffer[merge_idx] = self.calculate_similarity_between_clips(self.memory_long[merge_idx], self.memory_long[merge_idx+1])

            # print("<<<_update_long_memory>>> self.similarity_buffer: ", self.similarity_buffer)
            # print("<<<_update_long_memory>>> self.mergecnt_buffer: ", self.mergecnt_buffer)
            # print("<<<_update_long_memory>>> self.time_buffer: ", self.time_buffer)
            
            del clip1, clip2, merged_tokens, sim_scores, time_diffs, merge_cnts
            del sim_penalties, time_penalties, merge_penalties, total_penalties
        return

    def update_anomaly_pool(self, frame_features: torch.Tensor, pg_score: float):
        """
        Store compressed visual tokens + score for high-anomaly frames.

        Anomaly Pool provides a dedicated, score-curated visual reference for
        the LLM — functionally distinct from SFTW (which maintains a timeline).
        Pool tokens are inserted at [PEMF][SFTW][Pool][Now/RT-Anomaly] in
        get_memory_tokens().

        Eviction policy: when pool is full, the entry with the LOWEST score
        is removed (score-based eviction), ensuring the most anomalous frames
        are retained.

        Args:
            frame_features: [H*W, C] raw vision features to compress
            pg_score: anomaly score for this trigger
        """
        with torch.no_grad():
            compressed = self.merge_tokens(
                frame_features.unsqueeze(0),
                target_num_token=self.anomaly_pool_tokens
            )  # [1, anomaly_pool_tokens, C]

            self.anomaly_pool.append(compressed)
            self.anomaly_pool_scores.append(pg_score)

            # Score-based eviction: remove lowest-score entry when full
            while len(self.anomaly_pool) > self.anomaly_pool_max_size:
                min_idx = 0
                min_score = self.anomaly_pool_scores[0]
                for i in range(1, len(self.anomaly_pool_scores)):
                    if self.anomaly_pool_scores[i] < min_score:
                        min_score = self.anomaly_pool_scores[i]
                        min_idx = i
                del self.anomaly_pool[min_idx]
                del self.anomaly_pool_scores[min_idx]

    def get_anomaly_pool_tokens(self):
        """Return concatenated Anomaly Pool visual tokens. [1, N_pool, C] or None."""
        if len(self.anomaly_pool) == 0:
            return None
        return torch.cat(self.anomaly_pool, dim=1)

    def update(self, new_frame: torch.Tensor):
        """
        new_frame: [H * W, C]
        frame_idx: current frame index (int)
        return: [1, N, C] memory feature sequence
        """
        with torch.no_grad():
            new_frame_flat = new_frame.reshape(-1)  # [T, H*W*C]
            if self.last_frame_flat is not None:
                self.frame_sim_buffer.append(F.cosine_similarity(self.last_frame_flat, new_frame_flat, dim=0))
            self.last_frame_flat = new_frame_flat
            
            # APS: default anomaly score = 0.0 (normal frame)
            self.now_anomaly_scores.append(0.0)
            
            new_frame_tokens = self.merge_tokens(new_frame.unsqueeze(0), self.st_memory_tokens[0])  # [1, N0, C]
            self.memory_now.append(new_frame_tokens)
            
            # update memory immediately to keep memory usage fixed
            self._update_short_memory()
            self._update_event_split()
            self._update_long_memory()

    def update_with_anomaly_score(self, new_frame: torch.Tensor, anomaly_score: float = 0.0):
        """
        Update memory with anomaly score for APS (Anomaly Priority Score) protection.
        
        Same as update() but additionally tracks per-frame anomaly scores.
        When events are split into PEMF long memory, the event-level anomaly score
        (max of constituent frames) is used by APS to protect anomaly-containing
        events from being merged/compressed away.
        
        Args:
            new_frame: [H * W, C] vision features of the frame
            anomaly_score: anomaly score for this frame (e.g. fused PG+SF score)
                          0.0 for normal frames, higher for anomaly frames
        """
        with torch.no_grad():
            new_frame_flat = new_frame.reshape(-1)
            if self.last_frame_flat is not None:
                self.frame_sim_buffer.append(F.cosine_similarity(self.last_frame_flat, new_frame_flat, dim=0))
            self.last_frame_flat = new_frame_flat
            
            # APS: store anomaly score for this frame
            self.now_anomaly_scores.append(anomaly_score)
            
            new_frame_tokens = self.merge_tokens(new_frame.unsqueeze(0), self.st_memory_tokens[0])
            self.memory_now.append(new_frame_tokens)
            
            self._update_short_memory()
            self._update_event_split()
            self._update_long_memory()


    def get_anomaly_context(self) -> dict:
        """
        Get text-based anomaly context from the Anomaly Pool for prompt enrichment.
        
        Instead of injecting Anomaly Pool visual tokens into the image sequence
        (which breaks temporal ordering and pretrain compatibility), we extract
        statistical summaries that can be embedded in the TEXT prompt.
        
        Returns:
            dict with keys:
                - "pool_size": int, number of anomaly frames tracked
                - "peak_score": float, highest anomaly score in pool (0.0 if empty)
                - "avg_score": float, average anomaly score in pool (0.0 if empty)
                - "context_str": str, ready-to-use text for prompt injection
                    e.g. "4 anomaly alerts recorded (peak: 78%, avg: 52%)."
                    or "" (empty string) if pool is empty
        """
        pool_size = len(self.anomaly_pool_scores)
        if pool_size == 0:
            return {
                "pool_size": 0,
                "peak_score": 0.0,
                "avg_score": 0.0,
                "context_str": "",
            }
        peak_score = max(self.anomaly_pool_scores)
        avg_score = sum(self.anomaly_pool_scores) / pool_size
        context_str = (
            f"{pool_size} prior anomaly alert(s) recorded "
            f"(peak: {int(round(peak_score * 100))}%, avg: {int(round(avg_score * 100))}%)."
        )
        return {
            "pool_size": pool_size,
            "peak_score": peak_score,
            "avg_score": avg_score,
            "context_str": context_str,
        }

    def get_memory_tokens(self, rt_anomaly_tokens=None, include_now=True):
        """
        Build the full memory token sequence for LLM inference.
        
        Output order: [PEMF (Long)] [SFTW (Short)] [Anomaly Pool] [RT-Anomaly or Now]
        
        PEMF = oldest, most compressed (64 tok/frame)
        SFTW = recent timeline, moderately compressed (128 tok/frame)
        Pool = score-curated anomaly reference frames (128 tok/frame)
        Now  = current high-detail frame (729 tok, training) or
        RT-Anomaly = dense anomaly frames (4×729 tok, eval trigger)
        
        The Pool sits between SFTW and Now because:
        - It provides anomaly references that the LLM should attend to BEFORE
          the current frame (similar to a "contextual prior").
        - Its temporal position is approximate but functional: the LLM learns
          during finetuning that Pool tokens represent salient anomaly frames.
        
        Args:
            rt_anomaly_tokens: Optional [N_frames, H*W, C] tensor of real-time anomaly frames.
                              When provided, these replace Now at the end of the sequence.
            include_now: Whether to include memory_now in the output.
                        Automatically False when rt_anomaly_tokens is provided.
        
        Returns:
            [1, N_total, C] concatenated memory token sequence.
        """
        # If RT-Anomaly is provided, it replaces Now
        if rt_anomaly_tokens is not None:
            include_now = False
        
        with torch.no_grad():
            self._update_short_memory()
            self._update_event_split()
            self._update_long_memory()
            x_all = []
            
            # 1. PEMF (Long Memory) — oldest, most compressed (64 tok/frame)
            if len(self.memory_long) > 0:
                x_all.append(torch.cat(self.memory_long, dim=0).reshape(1, -1, self.hidden_size))
            
            # 2. SFTW (Short Memory) — recent, moderately compressed (128 tok/frame)
            if len(self.memory_short) > 0:
                x_all.append(torch.cat(self.memory_short, dim=0).reshape(1, -1, self.hidden_size))
            
            # 3. Anomaly Pool — score-curated anomaly reference tokens
            pool_tokens = self.get_anomaly_pool_tokens()
            if pool_tokens is not None:
                x_all.append(pool_tokens)
            
            # 4a. RT-Anomaly (dense anomaly frames, replaces Now)
            if rt_anomaly_tokens is not None:
                if rt_anomaly_tokens.dim() == 2:
                    # Single frame [H*W, C] → [1, H*W, C]
                    rt_anomaly_tokens = rt_anomaly_tokens.unsqueeze(0)
                if rt_anomaly_tokens.dim() == 3 and rt_anomaly_tokens.shape[0] != 1:
                    # Multiple frames [N, H*W, C] → [1, N*H*W, C]
                    rt_anomaly_tokens = rt_anomaly_tokens.reshape(1, -1, self.hidden_size)
                x_all.append(rt_anomaly_tokens)
            # 4b. Now (normal mode / training mode)
            elif include_now and len(self.memory_now) > 0:
                x_all.append(torch.cat(self.memory_now, dim=0).reshape(1, -1, self.hidden_size))

        return torch.cat(x_all, dim=1)  # [1, N_total, C]


