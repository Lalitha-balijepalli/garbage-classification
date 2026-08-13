"""
recycling.py
============
Smart Recycling Recommendation engine (Section 19).

Design note: the notebook's EDA step (Section 7 / CELL 9-10) discovers the
REAL class names from the downloaded dataset — do not assume them ahead of
time. The Kaggle "garbage-classification" dataset by asdasdasasdas has, as
of this writing, historically shipped six classes: cardboard, glass, metal,
paper, plastic, trash. RECYCLING_GUIDE below covers those six plus a
generic fallback entry, so the app still works if class names differ
slightly (e.g. capitalisation, or an extra "biological"/"battery" class
some dataset mirrors add). get_recommendation() normalises the lookup key
and falls back to a generic-but-honest response for anything unmapped,
rather than crashing or inventing specifics.
"""

RECYCLING_GUIDE = {
    "cardboard": {
        "recyclable": True,
        "disposal_method": "Dry, flattened cardboard goes in the paper/cardboard recycling stream.",
        "instructions": [
            "Remove any tape, staples, or plastic packaging.",
            "Flatten boxes to save space and help sorting machinery.",
            "Keep cardboard dry — wet or food-soiled cardboard (e.g. greasy pizza boxes) usually cannot be recycled and should go in general/organic waste instead.",
        ],
        "reuse_ideas": [
            "Storage boxes or drawer organisers",
            "Packing material for shipping",
            "Kids' craft projects or compost browns (shredded, uncoated cardboard)",
        ],
        "environmental_impact": (
            "Recycling cardboard saves trees, reduces landfill volume, and uses "
            "significantly less energy than producing new cardboard from raw pulp."
        ),
        "safety": "Generally safe to handle; watch for staples or sharp edges on box flaps.",
    },
    "glass": {
        "recyclable": True,
        "disposal_method": "Glass bottles and jars go in the glass recycling stream (check if your locality separates by colour).",
        "instructions": [
            "Empty and rinse out any food or liquid residue.",
            "Remove lids/caps — these are often a different material and recycled separately.",
            "Do not include broken window glass, mirrors, or ceramics — these have different melting points and contaminate glass recycling batches.",
        ],
        "reuse_ideas": [
            "Reusable storage jars",
            "Drinking glasses or vases",
            "Candle holders",
        ],
        "environmental_impact": (
            "Glass is 100% and infinitely recyclable without loss of quality. Recycled "
            "glass ('cullet') reduces furnace energy demand and raw material extraction."
        ),
        "safety": "Handle broken glass carefully; wrap sharp shards in paper/cardboard before disposal to protect waste handlers.",
    },
    "metal": {
        "recyclable": True,
        "disposal_method": "Aluminium and steel cans/containers go in the metal recycling stream.",
        "instructions": [
            "Rinse out food or drink residue.",
            "Cans can usually be left whole; large scrap metal may need a dedicated scrap/e-waste facility.",
            "Separate metal from any non-metal packaging attached to it.",
        ],
        "reuse_ideas": [
            "Pen or utensil holders",
            "Planters for small herbs",
            "Storage tins",
        ],
        "environmental_impact": (
            "Recycling aluminium uses about 95% less energy than producing new aluminium "
            "from ore, making it one of the most energy-efficient materials to recycle."
        ),
        "safety": "Watch for sharp cut edges on opened cans.",
    },
    "paper": {
        "recyclable": True,
        "disposal_method": "Clean, dry paper goes in the paper recycling stream.",
        "instructions": [
            "Remove plastic windows (e.g. from envelopes) where possible.",
            "Keep paper dry and free of food contamination.",
            "Shredded paper may need to be bagged separately depending on local rules.",
        ],
        "reuse_ideas": [
            "Scrap notepaper",
            "Packing/cushioning material",
            "Compost (uncoated, non-glossy paper only)",
        ],
        "environmental_impact": (
            "Recycling paper reduces deforestation and the energy/water used in producing "
            "virgin paper pulp."
        ),
        "safety": "No special precautions beyond general handling.",
    },
    "plastic": {
        "recyclable": True,
        "disposal_method": "Check the resin code (the number in the recycling triangle, e.g. PET/1, HDPE/2) — most curbside programs accept #1 and #2 plastics; others vary by locality.",
        "instructions": [
            "Empty and rinse the container.",
            "Remove non-plastic parts (e.g. pumps, metal springs) if separable.",
            "Separate the cap if your local program requires it.",
            "Place in the appropriate recyclable-waste bin per your local guidelines.",
        ],
        "reuse_ideas": [
            "Plant containers",
            "Storage containers",
            "DIY craft projects (e.g. bird feeders, organisers)",
        ],
        "environmental_impact": (
            "Plastic waste can persist in the environment for hundreds of years and "
            "contributes to pollution — including microplastics — when improperly disposed of."
        ),
        "safety": "Rinse containers that held chemicals before recycling or reuse; avoid reusing plastic that held toxic substances for food storage.",
    },
    "trash": {
        "recyclable": False,
        "disposal_method": "This item is best suited for general (non-recyclable) waste collection.",
        "instructions": [
            "Confirm it cannot be recycled or composted locally before discarding — some 'trash'-classified items are still accepted by specialised drop-off programs (e.g. textiles, batteries, electronics).",
            "Dispose of via your regular household waste collection.",
        ],
        "reuse_ideas": [
            "Consider whether the item can be repaired, donated, or repurposed before disposal.",
        ],
        "environmental_impact": (
            "Non-recyclable waste typically goes to landfill or incineration; reducing "
            "consumption of single-use, non-recyclable items is the most effective way to "
            "lower this impact."
        ),
        "safety": "If the item is sharp, chemical, electronic, or battery-containing, do NOT place it in general waste — take it to a hazardous-waste or e-waste facility instead.",
    },
    # Generic fallback — used only if the trained model's class names don't
    # match any key above (e.g. a dataset variant with extra categories).
    "_default": {
        "recyclable": None,
        "disposal_method": (
            "This category is not yet in the recycling knowledge base. "
            "Please check your local municipal waste guidelines for this material."
        ),
        "instructions": [
            "Search your city/municipality's official waste-sorting guide for this item.",
        ],
        "reuse_ideas": [],
        "environmental_impact": "Not available for this category.",
        "safety": "When unsure, treat unidentified waste cautiously and consult local guidelines.",
    },
}


def _normalize_key(class_name: str) -> str:
    return class_name.strip().lower().replace(" ", "_").replace("-", "_")


def get_recommendation(predicted_class: str) -> dict:
    """
    Look up the recycling recommendation for a predicted class name.
    Falls back to a generic, honest response rather than fabricating
    details for classes not in RECYCLING_GUIDE (per Section 20's
    instruction not to give unreliable disposal instructions).
    """
    key = _normalize_key(predicted_class)
    guide = RECYCLING_GUIDE.get(key, RECYCLING_GUIDE["_default"])
    return {"predicted_class": predicted_class, **guide}


def format_recommendation_text(predicted_class: str, confidence: float, recommendation: dict) -> str:
    """Human-readable text block matching the Section 20 example format."""
    lines = [
        f"Detected Waste: {predicted_class.capitalize()}",
        f"Confidence: {confidence * 100:.1f}%",
        "",
        f"Recyclability: {'Recyclable' if recommendation['recyclable'] else 'Not typically recyclable' if recommendation['recyclable'] is False else 'Unknown — check local guidelines'}",
        "",
        f"Recommended Action:\n{recommendation['disposal_method']}",
    ]
    if recommendation["instructions"]:
        lines.append("\nSteps:")
        lines += [f"{i+1}. {step}" for i, step in enumerate(recommendation["instructions"])]
    if recommendation["reuse_ideas"]:
        lines.append("\nReuse Ideas:")
        lines += [f"- {idea}" for idea in recommendation["reuse_ideas"]]
    lines.append(f"\nEnvironmental Impact:\n{recommendation['environmental_impact']}")
    lines.append(f"\nSafety:\n{recommendation['safety']}")
    return "\n".join(lines)
