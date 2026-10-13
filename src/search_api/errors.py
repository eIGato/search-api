"""Errors raised by the service layer. The API layer maps them to HTTP responses."""


class DomainError(Exception):
    @property
    def message(self) -> str:
        return str(self)


class NotFoundError(DomainError):
    def __init__(self, entity: str) -> None:
        super().__init__(f"{entity} not found")


class ConflictError(DomainError):
    pass


class InvalidRequestError(DomainError):
    pass
