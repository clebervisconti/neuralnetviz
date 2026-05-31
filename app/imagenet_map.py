"""Map MobileNetV2's 1000 ImageNet predictions down to the 6 animal classes
NeuraNetViz teaches (bird/cat/deer/dog/frog/horse).

Approach: keyword-match on the human-readable label that `decode_predictions`
returns. Each rule is an ordered tuple — a label is bucketed into the first
match. The match is a substring test over the lowercased label.

Why this approach (vs. hand-curating 1000 indices):
- Robust to model updates that re-order indices.
- Self-documenting: a teacher can read the rules and know exactly why
  "golden_retriever" → "dog".
- Edge cases (deer is not first-class in ImageNet; the closest is "hartebeest",
  "gazelle", "impala", "ibex" — we group those under "deer" since they're the
  same "hoofed mammal with antlers/horns" silhouette students will recognize).
"""
from __future__ import annotations

from typing import Iterable

# Order matters — first match wins.
RULES: list[tuple[str, tuple[str, ...]]] = [
    ("dog", (
        "dog", "retriever", "spaniel", "terrier", "hound", "poodle",
        "pug", "bulldog", "boxer", "schnauzer", "shepherd", "collie",
        "sheepdog", "pinscher", "doberman", "rottweiler", "mastiff",
        "chihuahua", "dachshund", "beagle", "labrador", "husky",
        "malamute", "samoyed", "pomeranian", "papillon", "shih-tzu",
        "shih_tzu", "newfoundland", "saint_bernard", "bernese",
        "great_dane", "great_pyrenees", "akita", "basenji", "chow",
        "corgi", "vizsla", "weimaraner", "whippet", "saluki",
        "borzoi", "afghan_hound", "wolfhound", "puppy", "papillon",
        "dingo",
    )),
    ("cat", (
        "tabby", "persian_cat", "siamese_cat", "egyptian_cat",
        "tiger_cat", "lynx", "ocelot",  # use lookalikes as cat for kids' demos
        "cougar", "leopard", "jaguar", "panther", "snow_leopard",
        "tiger", "lion", "cheetah",
        "cat",  # generic — last so above rules apply first
    )),
    ("bird", (
        "bird", "cock", "hen", "chicken", "ostrich", "brambling",
        "goldfinch", "finch", "junco", "bunting", "robin", "bulbul",
        "jay", "magpie", "chickadee", "ouzel", "kite", "eagle",
        "vulture", "owl", "hawk", "falcon", "parrot", "macaw",
        "cockatoo", "lorikeet", "hornbill", "hummingbird", "toucan",
        "duck", "goose", "swan", "pelican", "albatross", "flamingo",
        "spoonbill", "stork", "heron", "egret", "crane", "bittern",
        "ibis", "limpkin", "redshank", "ruddy_turnstone", "dowitcher",
        "oystercatcher", "peacock", "quail", "partridge", "grouse",
        "ptarmigan", "prairie_chicken", "sparrow", "wren", "kestrel",
        "kookaburra", "lorikeet", "vulture", "buzzard", "condor",
        "swallow", "starling", "crow", "raven", "penguin", "pheasant",
    )),
    ("frog", (
        "frog", "toad", "tree_frog", "bullfrog", "tailed_frog",
    )),
    ("horse", (
        "horse", "sorrel", "zebra", "pony",
    )),
    # Deer is not a first-class ImageNet category, but we let hoofed mammals
    # with antlers/horns count — students recognize the silhouette.
    ("deer", (
        "deer", "elk", "moose", "antelope", "gazelle", "impala",
        "hartebeest", "bison", "buffalo", "ox", "water_buffalo",
        "ibex", "bighorn",
    )),
]


def classify_label(label: str) -> str | None:
    """Return one of {'bird','cat','deer','dog','frog','horse'} or None."""
    l = label.lower().replace("-", "_")
    for cls, kws in RULES:
        for kw in kws:
            if kw in l:
                return cls
    return None


def aggregate(decoded: Iterable[tuple[str, str, float]], classes: list[str]) -> dict[str, float]:
    """Sum probabilities of decoded ImageNet predictions into our 6 buckets.

    `decoded` is the output of tf.keras.applications.*.decode_predictions —
    a list of (wnid, readable_name, probability) triples.
    """
    buckets = {c: 0.0 for c in classes}
    for _wnid, name, prob in decoded:
        cls = classify_label(name)
        if cls and cls in buckets:
            buckets[cls] += float(prob)
    total = sum(buckets.values())
    if total > 0:
        # normalize across the 6 classes — what's left over is "unrecognized
        # in our 6", which we surface implicitly by the absolute total below.
        buckets = {k: v / total for k, v in buckets.items()}
    return buckets
