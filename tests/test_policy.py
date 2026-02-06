from runtime.policy import sensitive_query_reason


def test_sensitive_detection() -> None:
    assert sensitive_query_reason("what is my OTP") == "otp"
    assert sensitive_query_reason("show my credit card") == "card_number"
    assert sensitive_query_reason("what did i say to pranav") is None
