import coreaudio as ca


def test_coreaudio():

	assert ca.test_error() == -4
