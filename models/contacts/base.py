"""Base contract for pairwise temporal-contact models."""

from base.model import Model


CONTACT_DATASET = "contacts"


class ContactModel(Model):
    """A model whose native simulation uses the ``t``, ``i``, ``j`` schema."""

    dataset = CONTACT_DATASET


__all__ = ["CONTACT_DATASET", "ContactModel"]
