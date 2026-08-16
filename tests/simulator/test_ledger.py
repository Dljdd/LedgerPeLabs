"""Behavioral coverage for the double-entry value ledger."""

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from apar.simulator.ledger import Ledger, LedgerEntry


@pytest.fixture
def ledger() -> Ledger:
    return Ledger()


def test_ledger_rejects_unbalanced_posting(ledger: Ledger) -> None:
    """Catch entries that create or destroy value at post time."""
    with pytest.raises(ValueError, match="debits must equal credits"):
        ledger.post(
            LedgerEntry("e1", debit={"payer": Decimal("10")}, credit={"payee": Decimal("9")})
        )


def test_ledger_quantizes_legs_to_declared_currency_exponent(ledger: Ledger) -> None:
    """Catch retaining fractional cents in USD postings."""
    ledger = Ledger(opening_balances={"payer": Decimal("2")})
    ledger.post(
        LedgerEntry(
            "usd-rounding",
            debit={"payer": Decimal("1.005")},
            credit={"payee": Decimal("1.005")},
        )
    )

    assert ledger.balance("payer") == Decimal("1.00")
    assert ledger.balance("payee") == Decimal("1.00")


@pytest.mark.parametrize(
    ("currency", "amount", "expected"),
    [
        ("JPY", Decimal("4.6"), Decimal("5")),
        ("KWD", Decimal("4.1236"), Decimal("4.124")),
    ],
)
def test_ledger_uses_currency_specific_exponents(
    currency: str, amount: Decimal, expected: Decimal
) -> None:
    """Catch applying USD cent precision to every currency."""
    ledger = Ledger(allow_credit={"payer"})
    ledger.post(LedgerEntry("precision", {"payer": amount}, {"payee": amount}, currency))

    assert ledger.balance("payer", currency) == -expected
    assert ledger.balance("payee", currency) == expected


def test_ledger_rejects_unsupported_currency(ledger: Ledger) -> None:
    """Catch guessing a precision for an unknown currency."""
    entry = LedgerEntry("unknown", {"payer": Decimal("1")}, {"payee": Decimal("1")}, "BTC")

    with pytest.raises(ValueError, match="unsupported currency"):
        ledger.post(entry)


def test_ledger_rejects_negative_posting_leg(ledger: Ledger) -> None:
    """Catch negative legs that invert debit or credit semantics."""
    entry = LedgerEntry("negative", {"payer": Decimal("-1")}, {"payee": Decimal("-1")})

    with pytest.raises(ValueError, match="non-negative"):
        ledger.post(entry)


def test_ledger_rejects_overdraft_without_credit_configuration(ledger: Ledger) -> None:
    """Catch spending from an unfunded account."""
    entry = LedgerEntry("overdraft", {"payer": Decimal("1")}, {"payee": Decimal("1")})

    with pytest.raises(ValueError, match="overdraw"):
        ledger.post(entry)


def test_ledger_accepts_explicit_opening_balances() -> None:
    """Catch ignoring funds supplied for later rail simulations."""
    ledger = Ledger(opening_balances={"payer": Decimal("10.00")})
    ledger.post(LedgerEntry("transfer", {"payer": Decimal("4")}, {"payee": Decimal("4")}))

    assert ledger.balance("payer") == Decimal("6.00")
    assert ledger.balance("payee") == Decimal("4.00")
    ledger.assert_conserved()


def test_ledger_conservation_accounts_for_declared_external_adjustments() -> None:
    """Catch conservation checks that silently ignore declared off-ledger value movement."""
    ledger = Ledger(
        opening_balances={"payer": Decimal("90.00")},
        external_adjustments={"USD": Decimal("10.00")},
    )
    ledger.post(LedgerEntry("transfer", {"payer": Decimal("4")}, {"payee": Decimal("4")}))

    ledger.assert_conserved()


def test_ledger_history_is_append_only(ledger: Ledger) -> None:
    """Catch callers mutating a posted entry or the returned history."""
    funded = Ledger(opening_balances={"payer": Decimal("5")})
    entry = LedgerEntry("transfer", {"payer": Decimal("1")}, {"payee": Decimal("1")})
    funded.post(entry)

    assert funded.entries == (entry,)
    with pytest.raises(AttributeError):
        funded.entries.append(entry)  # type: ignore[attr-defined]
    assert funded.entries == (entry,)


@given(
    opening=st.integers(min_value=0, max_value=10_000),
    transfers=st.lists(st.integers(min_value=0, max_value=10_000), min_size=1, max_size=20),
)
def test_ledger_conservation_holds_for_randomized_balanced_postings(
    opening: int, transfers: list[int]
) -> None:
    """Catch drift in balances after an arbitrary sequence of balanced postings."""
    total = sum(transfers)
    ledger = Ledger(opening_balances={"payer": Decimal(opening + total) / 100})

    for sequence, cents in enumerate(transfers):
        amount = Decimal(cents) / 100
        ledger.post(LedgerEntry(str(sequence), {"payer": amount}, {"payee": amount}))
        ledger.assert_conserved()

    assert ledger.balance("payer") + ledger.balance("payee") == Decimal(opening + total) / 100
