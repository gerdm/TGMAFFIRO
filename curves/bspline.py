import numpy as np

class BSpline(object):
    def __init__(self, knots, degree):
        self.knots = knots
        self.degree = degree

    def eval(self, t, knot, degree=None):
        """
        Evaluate the B-Spline at a given knot and t in the
        range of knots.
        
        Parameters
        ---------
        t:  float
            The point at which the B-Spline will be evaluated

        knot:  int
            The index of the knot to reference the bspline

        degree: int, optional
                The degree of the B-Spline

        Returns
        -------
        float
            The degree 'd' B-Spline evaluated at around the
            'k'-th node, with a value of 't'
        """
        if degree is None:
            degree = self.degree


        if degree == 0:
            degree0_spline = lambda t, k: (self.knots[k] <= t < self.knots[k + 1]) * 1
            degree0_spline = np.vectorize(degree0_spline)
            return degree0_spline(t, knot)
        else:
            # B_k^{(d-1)}(t)
            B_k_dm1_t = self.eval(t, knot, degree = degree-1)
            # B_{k+1}^{(d-1)}(t)
            B_kp1_dm1_t = self.eval(t, knot + 1, degree = degree-1)
            # Auxiliary values to comute the bspline
            term1 = (t - self.knots[knot]) / (self.knots[knot + degree] - self.knots[knot])
            term2 = (self.knots[knot + degree + 1] - t) / (self.knots[knot + degree + 1] - self.knots[knot + 1])

            return term1 * B_k_dm1_t + term2 * B_kp1_dm1_t
