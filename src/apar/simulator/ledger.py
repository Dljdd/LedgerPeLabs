"""An append-only, currency-aware double-entry ledger for simulations."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from types import MappingProxyType

CurrencyAmount = Mapping[str, Decimal]
AccountReference = str | tuple[str, str]

_CURRENCY_EXPONENTS: Mapping[str, int] = MappingProxyType({"EUR": 2, "JPY": 0, "KWD": 3, "USD": 2})


def _frozen_amounts(amounts: CurrencyAmount) -> CurrencyAmount:
    """Copy posting legs so a caller cannot revise a recorded entry in place."""
    return MappingProxyType(dict(amounts))


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """A requested same-currency debit and credit posting."""

    entry_id: str
    debit: CurrencyAmount
    credit: CurrencyAmount
    currency: str = "USD"

    def __post_init__(self) -> None:
        object.__setattr__(self, "debit", _frozen_amounts(self.debit))
        object.__setattr__(self, "credit", _frozen_amounts(self.credit))


class Ledger:
    """Track account balances through validated, append-only double-entry postings."""

    def __init__(
        self,
        opening_balances: Mapping[AccountReference, Decimal] | None = None,
        *,
        allow_credit: Collection[AccountReference] | None = None,
    ) -> None:
        self._allow_credit = frozenset(allow_credit or ())
        self._balances: dict[tuple[str, str], Decimal] = {}
        self._entries: list[LedgerEntry] = []
        self._entry_ids: set[str] = set()

        for account, amount in (opening_balances or {}).items():
            account_name, currency = self._parse_account_reference(account)
            normalized = self._quantize(amount, currency)
            if normalized < 0 and not self._credit_allowed(account_name, currency):
                raise ValueError(f"opening balance would overdraw account: {account_name}")
            self._balances[(account_name, currency)] = normalized

        self._opening_totals = self._totals_by_currency(self._balances)

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        """Return an immutable snapshot of the posting history."""
        return tuple(self._entries)

    def balance(self, account: str, currency: str = "USD") -> Decimal:
        """Return an account's current balance in one supported currency."""
        self._currency_exponent(currency)
        return self._balances.get((account, currency), Decimal(0))

    def post(self, entry: LedgerEntry) -> None:
        """Validate and atomically append a balanced posting."""
        if not entry.entry_id.strip():
            raise ValueError("entry_id must not be empty")
        if entry.entry_id in self._entry_ids:
            raise ValueError(f"duplicate ledger entry_id: {entry.entry_id}")
        currency = entry.currency
        self._currency_exponent(currency)
        debits = self._normalize_legs(entry.debit, currency)
        credits = self._normalize_legs(entry.credit, currency)
        debit_total = sum(debits.values(), Decimal(0))
        credit_total = sum(credits.values(), Decimal(0))
        if debit_total != credit_total:
            raise ValueError("debits must equal credits")

        new_balances = self._balances.copy()
        for account, amount in debits.items():
            key = (account, currency)
            new_balance = new_balances.get(key, Decimal(0)) - amount
            if new_balance < 0 and not self._credit_allowed(account, currency):
                raise ValueError(f"posting would overdraw account: {account}")
            new_balances[key] = new_balance
        for account, amount in credits.items():
            key = (account, currency)
            new_balances[key] = new_balances.get(key, Decimal(0)) + amount

        normalized_entry = LedgerEntry(entry.entry_id, debits, credits, currency)
        self._balances = new_balances
        self._entries.append(normalized_entry)
        self._entry_ids.add(entry.entry_id)

    def assert_conserved(self) -> None:
        """Raise if append-only postings have changed value in any currency."""
        current_totals = self._totals_by_currency(self._balances)
        currencies = set(current_totals) | set(self._opening_totals)
        for currency in currencies:
            current = current_totals.get(currency, Decimal(0))
            opening = self._opening_totals.get(currency, Decimal(0))
            if current != opening:
                raise AssertionError(f"ledger value is not conserved for currency: {currency}")

    def _normalize_legs(self, legs: CurrencyAmount, currency: str) -> dict[str, Decimal]:
        normalized: dict[str, Decimal] = {}
        for account, amount in legs.items():
            if not account:
                raise ValueError("account must not be empty")
            if not isinstance(amount, Decimal):
                raise TypeError("ledger amounts must be Decimal")
            if not amount.is_finite() or amount < 0:
                raise ValueError("posting legs must be finite and non-negative")
            value = self._quantize(amount, currency)
            normalized[account] = value
        return normalized

    def _parse_account_reference(self, account: AccountReference) -> tuple[str, str]:
        if isinstance(account, tuple):
            if len(account) != 2:
                raise ValueError(
                    "opening balance account reference must contain account and currency"
                )
            account_name, currency = account
        else:
            account_name, currency = account, "USD"
        if not account_name:
            raise ValueError("account must not be empty")
        self._currency_exponent(currency)
        return account_name, currency

    def _credit_allowed(self, account: str, currency: str) -> bool:
        return account in self._allow_credit or (account, currency) in self._allow_credit

    def _quantize(self, amount: Decimal, currency: str) -> Decimal:
        exponent = self._currency_exponent(currency)
        if not isinstance(amount, Decimal):
            raise TypeError("ledger amounts must be Decimal")
        if not amount.is_finite():
            raise ValueError("ledger amounts must be finite")
        return amount.quantize(Decimal(1).scaleb(-exponent), rounding=ROUND_HALF_EVEN)

    @staticmethod
    def _totals_by_currency(balances: Mapping[tuple[str, str], Decimal]) -> dict[str, Decimal]:
        totals: dict[str, Decimal] = {}
        for (_, currency), amount in balances.items():
            totals[currency] = totals.get(currency, Decimal(0)) + amount
        return totals

    @staticmethod
    def _currency_exponent(currency: str) -> int:
        try:
            return _CURRENCY_EXPONENTS[currency]
        except KeyError as error:
            raise ValueError(f"unsupported currency: {currency}") from error
