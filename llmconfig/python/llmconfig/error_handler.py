from typing import Sequence

from fastapi.exceptions import RequestValidationError, RequestErrorModel
# from pydantic.error_wrappers import ErrorList

# class ParameterNotExistsError(BaseException):
#     def __init__(self, errors: Sequence[ErrorList]) -> None:
#         super().__init__(errors)

class ParameterNotExistsError(BaseException):
    """ Unspecified run-time error. """
    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass