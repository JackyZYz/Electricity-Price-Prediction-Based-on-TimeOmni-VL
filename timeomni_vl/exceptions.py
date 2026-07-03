class TimeOmniVLError(Exception):
    pass


class DataLoadError(TimeOmniVLError):
    pass


class DataAlignmentError(TimeOmniVLError):
    pass


class BiTSIError(TimeOmniVLError):
    pass


class BackboneError(TimeOmniVLError):
    pass


class TrainingError(TimeOmniVLError):
    pass


class InferenceError(TimeOmniVLError):
    pass


class ConfigurationError(TimeOmniVLError):
    pass
