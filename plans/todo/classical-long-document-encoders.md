# Classical encoders for long documents

**Stage:** todo — needs a build contract before implementation · **Priority:** small, two days · **Home:** NOT Panelary — belongs with the filings pipeline

> Parked here only so it is not lost. It concerns document embedding for
> filings and a nightly inference server, neither of which is in this repo.
> Move it to the repository that owns that pipeline.

## Pitch

BeyondArena's own ablation: on long text TF-IDF generally beats
Qwen3-Embedding-8B, with the neural option winning 6% of comparisons; on short
text it wins 78%. The 50-character split was chosen for the ablation, not
derived, so the crossover is uncharacterised.

Filings are long text. The prior from the strongest available evidence is that
TF-IDF plus SVD, or a Model2Vec static table at up to 500x CPU speed, beats an
8B encoder on your corpus at a fraction of the cost — and deletes a GPU
dependency from a pipeline that runs nightly.

First experiment: one downstream task, three encoders: TF-IDF plus SVD,
potion-base-32M, Qwen3-8B. If the classical options win, decommission an
inference server.

## Assessment

Low risk, concrete payoff. One addition: sweep document length as a variable
rather than using a fixed split, since the crossover is the uncharacterised
part. Truncating filings at several lengths and re-running the three encoders
characterises the crossover on your own corpus instead of inheriting a
threshold chosen for someone else's ablation.
