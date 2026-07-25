# Where Intent Forms: Layer-Level Evidence of Injection Resistance via J-Lens

Team members: Alexander Christinck, Friedrich Herlten, Karin Riklin

**Project summary**

We take a single open-weight base model and create a fine-tuned variant trained specifically to resist prompt injection and jailbreaking (a lightweight fine-tune, e.g. LoRA on a curated set of injection/refusal examples, not a full retrain, to keep this feasible in the timeline). Both models receive the same system prompt containing classified information they're told never to reveal. We run both against a curated set of injection prompts of varying sophistication, drawn from an injection prompt database, and use J-lens to watch each model's internal concept space turn by turn as it processes each attempt.

For each prompt, we identify the layer(s) where the model's internal representation shifts from treating the injected content as "external text" to treating it as "instruction to act on," and compare this shift, whether it happens, how strongly, and how late, between the base model and the fine-tuned model. We also compare how often each model actually leaks the classified information. The core question isn't just whether fine-tuning improves security (that's measurable from leak rate alone), it's whether that improvement is visible internally in J-space, and if so, what it looks like: a delayed shift, a weaker shift, or no shift at all.

**Research question:** Does fine-tuning a model to resist prompt injection actually change its internal behavior, not just its output, and if so, how is that reflected in J-lens? Specifically, does the "external content to adopted intent" shift happen less often, later, or more weakly in the fine-tuned model compared to the base model?

**Hypothesis:** The fine-tuned model will leak the classified information less often than the base model across the same set of injection prompts, and this improved resistance will be visible in J-space: the fine-tuned model will show the "external content to adopted intent" shift less frequently, at a later layer, or with a weaker signal than the base model on the same prompts. In other words, fine-tuning for security doesn't just suppress the harmful output, it measurably changes when and whether the model internally treats injected content as its own intent.

