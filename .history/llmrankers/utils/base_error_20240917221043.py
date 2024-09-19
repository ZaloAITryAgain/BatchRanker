from http import HTTPStatus

from fastapi import HTTPException


class BaseError:
    def __init__(self, status: HTTPStatus, code: int, message: str):
        self.status = status
        self.code = code
        self.message = message

    def get_code(self):
        return self.code

    def get_message(self):
        return self.build_message()

    def get_status_code(self):
        return self.status.value

    def as_exception(self, custom_message=None):
        return ClientException(
            http_status_code=self.status.value,
            code=self.code,
            message=self.build_message(custom_message),
        )

    def as_http_exception(self, custom_message=None):
        return HTTPException(
            status_code=self.status.value,
            detail=self.build_message(custom_message),
        )

    def build_message(self, message=None):
        return message or self.message


class ClientException(Exception):
    def __init__(self, http_status_code: int, code: int, message: str):
        self.http_status_code = http_status_code
        self.code = code
        self.message = message
        super().__init__(message)
