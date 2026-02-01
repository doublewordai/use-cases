# Vulnerability Classification at Scale

## Summary

We tested whether LLMs can classify security vulnerabilities by type using 600 real-world vulnerable code samples from CVEfixes. Key findings:

1. **Qwen3-235B achieves 60% accuracy** on 3-way classification (Memory Safety / Integer / Pointer) - nearly 2x the random baseline
2. **Qwen significantly outperforms GPT-5** (60% vs 39%) on this security task
3. **Ensemble agreement predicts accuracy** - when 4 models agree, accuracy jumps to 68.5%; when they disagree, it's near-random
4. **62% of samples can be auto-triaged** based on model agreement, flagging only ambiguous cases for human review

## The Dataset

**CVEfixes** contains 138,000+ functions extracted from real vulnerability-fixing commits. Each sample is labeled with CWE types. We selected 600 samples across 6 CWEs:

| CWE | Description | Category |
|-----|-------------|----------|
| CWE-119 | Buffer Overflow | Memory Safety |
| CWE-125 | Out-of-bounds Read | Memory Safety |
| CWE-787 | Out-of-bounds Write | Memory Safety |
| CWE-190 | Integer Overflow | Integer |
| CWE-416 | Use After Free | Pointer |
| CWE-476 | NULL Pointer Dereference | Pointer |

## The Task

Given vulnerable C/C++ code, classify which vulnerability type it contains. The model sees 6 options and must pick one.

**Prompt:**
```
Analyze this C/C++ code for security vulnerabilities.
This code is known to contain a vulnerability. Identify what type.

Categories:
- CWE-125: Out-of-bounds Read
- CWE-787: Out-of-bounds Write
- CWE-119: Buffer Overflow
- CWE-190: Integer Overflow
- CWE-476: NULL Pointer Dereference
- CWE-416: Use After Free

Code: [vulnerable function]

Respond with JSON: {"cwe": "CWE-XXX", "confidence": "high/medium/low", "reasoning": "..."}
```

## Results

### Fine-grained Classification (6 classes)

| Model | Accuracy | Random Baseline |
|-------|----------|-----------------|
| Qwen3-235B | 26.3% | 16.7% |
| Qwen3-30B | 25.9% | 16.7% |
| GPT-5.2 | 25.2% | 16.7% |
| GPT-5-mini | 21.2% | 16.7% |

All models are ~1.5x better than random. Fine-grained CWE classification is hard - even frontier models struggle to distinguish between related vulnerability types.

### Grouped Classification (3 categories)

When we collapse related CWEs into broader categories:

| Model | Accuracy | Memory Safety | Integer | Pointer |
|-------|----------|---------------|---------|---------|
| **Qwen3-235B** | **60.0%** | 75.0% | 11.0% | 62.0% |
| Qwen3-30B | 55.1% | 73.0% | 8.3% | 52.1% |
| GPT-5.2 | 39.2% | 35.7% | 18.0% | 55.0% |
| GPT-5-mini | 36.3% | 40.7% | 12.0% | 42.0% |
| *Random* | *33.3%* | - | - | - |

**Key findings:**

1. **Qwen significantly outperforms GPT-5** - 60% vs 39% on grouped classification
2. **Memory safety detection is strong** - Qwen achieves 75% accuracy on buffer overflow variants
3. **Integer overflow is hard for everyone** - 8-18% across all models
4. **Pointer issues are moderate** - 52-62% for Qwen, 42-55% for GPT

### Top Confusions

Models consistently confuse related vulnerability types:

| Actual | Predicted As | Count |
|--------|--------------|-------|
| CWE-416 (Use After Free) | CWE-476 (NULL Deref) | 60 |
| CWE-190 (Integer Overflow) | CWE-476 (NULL Deref) | 42 |
| CWE-787 (OOB Write) | CWE-125 (OOB Read) | 40 |
| CWE-125 (OOB Read) | CWE-119 (Buffer Overflow) | 35 |

This makes sense - these are genuinely related vulnerability classes that share similar code patterns.

## Failure Examples

To understand why classification is hard, here are three representative failures:

### Example 1: Qwen Right, GPT Wrong

**CVE-2020-28097** - Linux VGA console scrollback buffer
**Actual:** CWE-125 (Out-of-bounds Read)

```c
static void vgacon_scrolldelta(struct vc_data *c, int lines)
{
    int start, end, count, soff;
    // ...
    soff = vgacon_scrollback_cur->tail -
        ((vgacon_scrollback_cur->cnt - end) * c->vc_size_row);
    soff -= count * c->vc_size_row;

    if (soff < 0)
        soff += vgacon_scrollback_cur->size;

    count = vgacon_scrollback_cur->cnt - start;
    // ... later uses soff as index into data buffer
}
```

**Qwen-235B (correct):**
> "The code uses 'soff' as an index without validating it stays within bounds. The adjustment may not guarantee it stays within the allocated buffer size, leading to reading from memory outside the allocated buffer."

**GPT-5.2 (wrong):**
> "A negative computed scrollback offset can be used as an index, causing out-of-bounds memory access." *(Predicted CWE-787 Out-of-bounds Write)*

**Why GPT failed:** Both models identified the bounds issue, but GPT incorrectly classified it as a *write* vulnerability when the actual CVE is about *reading* beyond buffer boundaries. Qwen correctly identified CWE-125 (read) vs GPT's CWE-787 (write).

---

### Example 2: GPT Right, Qwen Wrong

**CVE-2018-12896** - Linux kernel timer overflow
**Actual:** CWE-190 (Integer Overflow)

```c
void common_timer_get(struct k_itimer *timr, struct itimerspec64 *cur_setting)
{
    const struct k_clock *kc = timr->kclock;
    // ...
    if (iv && (timr->it_requeue_pending & REQUEUE_PENDING || sig_none))
        timr->it_overrun += (int)kc->timer_forward(timr, now);  // <- overflow here
    // ...
}
```

**Qwen-235B (wrong):**
> "The code dereferences 'timr->kclock' without checking if 'timr' or 'timr->kclock' is NULL." *(Predicted CWE-476)*

**GPT-5.2 (correct):**
> "The result of kc->timer_forward() is cast to int and added to timr->it_overrun without bounds checking, allowing overflow/wraparound of the overrun counter."

**Why Qwen failed:** It defaulted to the "missing NULL check" pattern - a common vulnerability but not the issue here. GPT correctly identified the integer overflow from the unchecked cast and addition.

---

### Example 3: Both Models Wrong

**CVE-2016-3841** - Linux kernel DCCP IPv6 use-after-free
**Actual:** CWE-416 (Use After Free) | **Both predicted:** CWE-476 (NULL Pointer)

```c
static int dccp_v6_send_response(const struct sock *sk, struct request_sock *req)
{
    struct inet_request_sock *ireq = inet_rsk(req);
    struct ipv6_pinfo *np = inet6_sk(sk);
    struct sk_buff *skb;
    struct dst_entry *dst;
    // ...
    dst = ip6_dst_lookup_flow(sk, &fl6, final_p);
    if (IS_ERR(dst)) {
        err = PTR_ERR(dst);
        dst = NULL;
        goto done;
    }
    skb = dccp_make_response(sk, dst, req);
    // ...
}
```

**Qwen-235B:**
> "The code dereferences 'np' without checking if it is NULL. If 'inet6_sk(sk)' returns NULL, then 'np->opt' will cause a NULL pointer dereference."

**GPT-5.2:**
> "The function may call dst_release(dst) even when dst is NULL, causing a NULL pointer dereference."

**Why both failed:** The real vulnerability is that `dst` can be used after being freed in a race condition - a use-after-free, not a NULL pointer issue. Both models focused on potential NULL dereferences rather than the subtle lifetime/concurrency issue.

---

### Takeaway

These examples reveal common failure modes:

1. **NULL pointer bias** - Models over-predict CWE-476 because "check for NULL" is such a common code review pattern
2. **Read/Write confusion** - The code patterns for OOB reads and writes are nearly identical; distinguishing them requires understanding data flow direction
3. **Lifetime issues are subtle** - Use-after-free bugs require reasoning about object lifetimes across function calls, which is harder than spotting missing NULL checks

## Cheap Calibration: Run Twice for Confidence

Doubleword's batch pricing is low enough that you can run the same model multiple times for calibration - and still pay less than a single GPT-5.2 run.

### The Approach

Run Qwen-235B twice on each sample. When both runs agree, you have higher confidence. When they disagree, flag for human review.

### Results (Grouped 3-way Classification)

| Signal | Samples | Accuracy |
|--------|---------|----------|
| High confidence + runs agree | 417 (71%) | **61.4%** |
| High confidence + runs *disagree* | 21 (4%) | **23.8%** |
| Medium/low confidence | 148 (25%) | 55.4% |

The key insight: **when high-confidence runs disagree, accuracy drops from 61% to 24%**. That's the calibration signal - disagreement catches cases where the model is confidently wrong.

### Cost Comparison

| Approach | Cost | Accuracy |
|----------|------|----------|
| 2× Qwen-235B runs | **$0.28** | 61% (on agreed samples) |
| 1× GPT-5.2 run | $1.05 | 39% |

Two Qwen runs cost **73% less** than one GPT run, while achieving better accuracy and providing a calibration signal.

### Triage Workflow

| Confidence | Definition | Samples | Accuracy | Action |
|------------|------------|---------|----------|--------|
| High | Both runs agree + high confidence | 417 (71%) | 61.4% | Auto-classify |
| Review | Runs disagree OR low confidence | 169 (29%) | ~45% | Human review |

**71% of samples can be auto-triaged** with the double-run approach. The 29% flagged for review includes the genuinely ambiguous cases - and catches overconfident errors.

This is the "more is different" insight: running the same cheap model twice doesn't make it smarter, but it reveals **which predictions to trust**.

## Cost Analysis

*Note: We used OpenAI's real-time API for implementation convenience (their batch API doesn't support partial results and expires batches after 24 hours). Cost comparisons below use batch pricing for all providers to ensure a fair comparison.*

| Provider | Model | Mode | 600 Samples | Cost |
|----------|-------|------|-------------|------|
| Doubleword | Qwen3-235B | Batch | ~500K tokens | ~$0.30 |
| Doubleword | Qwen3-30B | Batch | ~500K tokens | ~$0.10 |
| OpenAI | GPT-5.2 | Batch | ~500K tokens | ~$4.00 |
| OpenAI | GPT-5-mini | Batch | ~500K tokens | ~$0.20 |

Qwen3-235B delivers the best accuracy at 1/13th the cost of GPT-5.2.

## What We Learned

1. **Fine-grained CWE classification is hard** - 6-way classification barely beats random across all models

2. **Broader categories work better** - Grouping into Memory Safety / Integer / Pointer yields 60% accuracy

3. **Qwen outperforms GPT-5 on security** - 60% vs 39% on grouped classification, possibly due to training data differences

4. **Ensemble agreement = confidence** - When 4 models agree, accuracy is 68.5%. When they disagree, it's near-random. This enables automatic triage.

5. **More is different** - Running 4 models doesn't improve accuracy much, but it reveals which predictions to trust. 62% of samples can be auto-classified based on agreement.

6. **Batch pricing enables exploration** - Testing 4 models on 600 samples cost ~$9 total

## Limitations

1. **Label quality** - CVEfixes labels come from commit metadata, not expert annotation
2. **Related CWEs** - Some vulnerabilities legitimately belong to multiple categories
3. **Code context** - Snippets may be too short to fully determine vulnerability type
4. **Model versions** - Results may vary with model updates

## Reproducing This Experiment

```bash
cd bug-detection-ensemble && uv sync

# Set API keys
export DOUBLEWORD_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Run classification (batch - cheaper)
uv run bug-ensemble classify -n 100 -m 235b -o results/classify_235b

# Or realtime (faster)
uv run bug-ensemble classify-realtime -n 100 -m gpt5.2 -o results/classify_gpt52

# Analyze results
uv run bug-ensemble classify-analyze -o results/classify_235b
```

---

*Data: CVEfixes v1.0.7, 600 C/C++ functions across 6 CWE types. Models: Qwen3-30B, Qwen3-235B (Doubleword), GPT-5-mini, GPT-5.2 (OpenAI). All costs shown use batch API pricing.*
