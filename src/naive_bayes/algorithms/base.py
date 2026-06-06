from sklearn.base import BaseEstimator


class Algorithm(BaseEstimator):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        """Name of the object's class."""
        return self.__class__.__name__

    @property
    def identifier(self):
        """Name of the object.

        Name is made by combining the class name with the parameters
        passed at construction time.

        Constructed by recreating the initialisation call.
        Example: ``Algorithm(param_1=value)``
        """
        paramstring = ",".join((f"{k}={v}" for k, v in self.get_params().items()))
        return self.name + "(" + paramstring + ")"

    def __str__(self):
        return self.name
