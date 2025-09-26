package digest

import "math"

func IsComplexEqual(a, b complex128) bool {
	return isFloatEqual(real(a), real(b)) && isFloatEqual(imag(a), imag(b))
}

func isFloatEqual(a, b float64) bool {
	return a == b || math.Abs(a-b) < 1e-6
}
