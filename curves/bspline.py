import numpy as np

class BSpline(object):
    def __init__(self, knots, degree):
        self.knots = knots
        self.len_knots = len(knots)
        self.degree = degree

    def keval(self, t, knot, degree=None, out_bounds_eval=False):
        """
        Evaluate the B-Spline at a given knot and t in the
        range of knots. Optional to evaluate the bspline at
        a different degree as the one assigned at the initializiation
        
        Parameters
        ---------
        t:  float
            The point at which the B-Spline will be evaluated

        knot:  int
            The index of the knot to reference the bspline

        degree: int, optional
                The degree of the B-Spline

        out_bounds_eval:bool, optional
                        If True, it returns 0 for bspline evaluation out
                        of the possible bounds

        Returns
        -------
        float
            The degree 'd' B-Spline evaluated at around the
            'k'-th node, with a value of 't'
        """
        if degree is None:
            degree = self.degree

        if knot + degree + 1  >= self.len_knots:
            if out_bounds_eval:
                return 0
            else:
                err = ("BSpline cannot be evaluated for "
                       "{knot} + {degree} + 1 >= {lenkn}"
                       .format(knot=knot, degree=degree,
                       lenkn=self.len_knots))
                raise IndexError(err)

        if degree == 0:
            degree0_spline = lambda t, k: (self.knots[k] <= t < self.knots[k + 1]) * 1
            degree0_spline = np.vectorize(degree0_spline)
            return degree0_spline(t, knot)
        else:
            # B_k^{(d-1)}(t)
            B_k_dm1_t = self.keval(t, knot, degree = degree-1)
            # B_{k+1}^{(d-1)}(t)
            B_kp1_dm1_t = self.keval(t, knot + 1, degree = degree-1)
            # Auxiliary values to comute the bspline
            term1 = (t - self.knots[knot]) / (self.knots[knot + degree] - self.knots[knot])
            term2 = (self.knots[knot + degree + 1] - t) / (self.knots[knot + degree + 1] - self.knots[knot + 1])

            return term1 * B_k_dm1_t + term2 * B_kp1_dm1_t

    def integrate(self, knot, limsup, liminf=-np.Inf):
        """
        Evalute the integral of the B-Spline from liminf to limsup

        Parameters
        ---------
        knot:   int
                The knot to integrate the bspline
        limsup: float
                The upper limit of the integral
        liminf: float, optional
                The lower limit of the integral (default to enp.Inf)
            
        Returns
        -------
        float
            The integration of the bspline from liminf to limsup
        """
        if liminf != -np.Inf:
            lower_integral = self.integrate(knot, liminf)
            upper_integral = self.integrate(knot, limsup)
            return upper_integral - lower_integral
        else:
            k = int(knot)
            total_integration = 0
            while k + self.degree + 3 <= self.len_knots:
                kd1 = k + self.degree + 1
                constant = (self.knots[kd1] - self.knots[k]) / (self.degree + 1)
                spline_val = self.keval(limsup, k, degree=self.degree + 1)
                total_integration += constant * spline_val
                k += 1
            return total_integration


    def diff(self, t, knot):
        """
        Evaluate the derivative of the integral

        Paramters
        ---------
        t:  float
            The point at which the B-Spline will be evaluated

        knot:  int
            The index of the knot to reference the bspline


        Returns
        -------
        float
            The derivative of the B-Spline evaluated at t
        """
        degree_minus1 = self.degree - 1
        # B_k^{(d-1)}(t)
        B_k_dm1_t = self.keval(t, knot, degree=degree_minus1)
        # B_{k+1}^{(d-1)}(t)
        B_kp1_dm1_t = self.keval(t, knot+1, degree=degree_minus1)

        term1 = self.degree / (self.knots[knot + self.degree] - self.knots[knot])
        term2 = self.degree / (self.knots[knot + self.degree + 1] - self.knots[knot + 1])

        return term1 * B_k_dm1_t - term2 * B_kp1_dm1_t

