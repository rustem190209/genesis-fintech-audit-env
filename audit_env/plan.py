"""План открытых эпизодов: 10 train и 10 dev. Закрытые seed и сочетания параметров есть только у организаторов."""

ONE = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_error": 1, "period_cutoff": 1}
TWO = {t: 2 for t in ONE}
BOTH = ["fx_boundary_partial", "dup_cutoff"]


def _e(seed, name, **config):
    return {"seed": seed, "name": name, "config": config}


EPISODES = {
    "train": [
        _e(101, "clean_small", n_invoices=40, n_journal=80),
        _e(102, "clean_large", n_invoices=60, n_journal=600),
        _e(103, "duplicates", errors={"duplicate_invoice": 2}, n_journal=120),
        _e(104, "allocation", errors={"payment_allocation": 2}, n_journal=120),
        _e(105, "fx", errors={"fx_error": 2}, n_journal=120),
        _e(106, "cutoff", errors={"period_cutoff": 2}, n_journal=120),
        _e(107, "all_four", errors=ONE, n_journal=200),
        _e(108, "all_four_x2", errors=TWO, n_journal=300),
        _e(109, "interacting", interactions=BOTH, n_journal=200),
        _e(110, "mixed_large", errors=ONE, interactions=BOTH, n_invoices=70, n_journal=700),
    ],
    "dev": [
        _e(201, "clean_a", n_invoices=50, n_journal=100),
        _e(202, "clean_b", n_invoices=45, n_journal=400, base_currency="KZT"),
        _e(203, "duplicates", errors={"duplicate_invoice": 3}, n_journal=150),
        _e(204, "allocation", errors={"payment_allocation": 3}, n_journal=150, base_currency="EUR"),
        _e(205, "fx", errors={"fx_error": 3}, n_journal=150),
        _e(206, "cutoff", errors={"period_cutoff": 3}, n_journal=150),
        _e(207, "all_four", errors=ONE, n_journal=250, base_currency="USD"),
        _e(208, "all_four_x3", errors={t: 3 for t in ONE}, n_journal=400),
        _e(209, "interacting", interactions=BOTH, n_journal=250, base_currency="KZT"),
        _e(210, "mixed", errors={"fx_error": 2, "period_cutoff": 1}, interactions=["fx_boundary_partial"],
           n_invoices=65, n_journal=500),
    ],
}


def all_episodes():
    """Список (split, номер, запись плана) в стабильном порядке."""
    return [(split, i, item) for split, items in EPISODES.items() for i, item in enumerate(items, 1)]
