"""Show the per-token scores behind the AWS-key split in viterbi_vs_argmax_output.txt."""
import math, torch, torch.nn.functional as F
from opf import OPF

TEXT = "my pw is Tr0ub4dor&3 and the backup key is AKIA4EXAMPLE7QXZ2LMN, born 03/07/91 in Busan"
opf = OPF(device="cpu")
rt = opf.get_runtime()
li = rt.label_info
def tag(k):
    return "O" if k == 0 else f"{li.token_boundary_tags[k]}-{li.span_class_names[li.token_to_span_label[k]]}"

ids = rt.encoding.encode(TEXT)
with torch.no_grad():
    lp = F.log_softmax(rt.model(torch.tensor([ids], dtype=torch.int32),
                                attention_mask=torch.ones(1, len(ids), dtype=torch.bool)).float(), -1)[0]
toks = [rt.encoding.decode([t]) for t in ids]
start = toks.index(" AK")
end = toks.index(",", start)
print("top-3 labels per token (probability):")
for i in range(start, end + 1):
    p, k = lp[i].exp().topk(3)
    print(f"  {toks[i]!r:9s}", "  ".join(f"{tag(int(kk)):20s}{float(pp):.3f}" for pp, kk in zip(p, k)))

# Compare the two candidate endings for the last key token.
i = end - 1
lab = {name: idx for idx, name in ((k, tag(k)) for k in range(lp.shape[1]))}
print(f"\nlast key token {toks[i]!r}:")
for name in ["E-secret", "E-account_number", "I-secret", "O"]:
    print(f"  P({name:18s}) = {math.exp(float(lp[i, lab[name]])):.3f}")
prev = i - 1
print(f"previous token {toks[prev]!r}: argmax = {tag(int(lp[prev].argmax()))}")

dec = opf._get_decoder(rt, opf._resolve_effective_decoder_config())
ts = dec._transition_scores
print("\nViterbi transition scores (illegal = -1e9):")
for a, b in [("I-secret", "E-secret"), ("I-secret", "E-account_number")]:
    print(f"  {a} -> {b}: {float(ts[lab[a], lab[b]]):.0f}")

# Score whole candidate paths over the key tokens (sum of log-probs; legal transitions add 0).
key = range(start, end)
def path(kind, last):
    tags = [f"B-{kind}"] + [f"I-{kind}"] * (len(key) - 2) + [last]
    return sum(float(lp[j, lab[t]]) for j, t in zip(key, tags)), tags
print("\npath log-scores over the key tokens:")
for name, (sc, tags) in {
    "all secret (B..I..E-secret)": path("secret", "E-secret"),
    "all account (B..I..E-account_number)": path("account_number", "E-account_number"),
    "argmax mix (B..I-secret, E-account_number)": path("secret", "E-account_number"),
}.items():
    legal = tags[-1].split("-", 1)[1] == tags[0].split("-", 1)[1]
    print(f"  {name:44s} {sc:8.3f}  {'legal' if legal else 'ILLEGAL (-1e9 transition)'}")
