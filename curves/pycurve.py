import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# TODO: * Add computation of discount factor given a date
#       * Add computation of forward factor: F(t_k-1, t_k-1)
#       * Add simple, compound and continous conventions; market conventions

class RateCurves(object):
    def __init__(self, val_date, market_coupons,
                 number_flows, coupon_length, holidays=None):
        """
        Class to construct an interest rate curve lattice using market swap quotations
        :parameters:
        :val_date: The reference date to construct the curve
        :holidays: A numpy array list of datetime objects used to indicate when a weekday is not a
                   workday
        :pillars: 
        """
        self.val_date = val_date
        self.number_flows = number_flows
        self.coupon_length = coupon_length
        self.holidays = holidays
        self.pillars = self.compute_pillars()
        self.market_coupons = market_coupons
        self.discount_rates = np.zeros(len(pillars))
        self.days_to_pillar = np.array([(pillar - self.val_date).days
                                        for pillar in  self.pillars])
        self.market_flows = dict()
        self.fitted_curve = False
        self.rate_curve = None
        
    def compute_pillars(self):
        """
        Compute the maturity date for each of the market nx1 swaps in 
        the market
        """
        horizons = self.number_flows * self.coupon_length
        pillars = []
        for horizon in horizons:
            end_date = self.forward_busdays(int(horizon))
            pillars.append(end_date)
        
        return pillars
    
    def forward_busdays(self, days):
        """
        Return the date corresponding to the start date
        plus a given number of days. If the end date is either a weekend
        or a holiday, move forward one day until a workday is found
        """
        end_date = self.val_date + timedelta(days)
        weekday = end_date.weekday()

        if ((self.holidays is not None) and (end_date in self.holidays)) \
           or (weekday in [5,6]):
            return forward_busdays(end_date, 1, self.holidays)
        else:
            return end_date
    
    def length_flows(self, nflows):
        """
        Retrieve the start, end and number of days of "n" flows.
        """
        start_periods = np.array([forward_busdays(self.val_date,
                                  self.coupon_length * period, holidays)
                                  for period in range(nflows)])
        
        end_periods = np.array([forward_busdays(self.val_date,
                                self.coupon_length * period, holidays)
                                for period in range(1, nflows + 1)])

        period_days = np.array([(end - start).days for
                                start, end in zip(start_periods, end_periods)])

        return period_days
    
    # TODO: Add market conventions to compute the coupons 
    #       and the discount factor
    def estimate_bond_price(self, nth_rate, nflows):
        """
        Price a bond paying n fixed coupons + principal where
        the nth rate is unkown. The price of this bond should be 1,
        and the approximation to 1 is given by the input rate.
        This function is necessary to find the nth rate such that
        makes the value of bond 1.
        """
        rates_nx1 = self.discount_rates.copy()
        
        # Length of each coupon to pay
        coupons_len = self.length_flows(nflows)
        
        # Find the index to replace the rate
        rate_index = np.where(self.number_flows == nflows)[0]
        if len(rate_index) == 0:
            raise ValueError("The structure lattice has no {}x1 instrument".format(nflows))
        
        swap_coupon = self.market_coupons[rate_index]
        # Modifying the rates
        rates_nx1[rate_index] = nth_rate
        lattice = interp1d(self.days_to_pillar, rates_nx1)
        discount_rates = lattice(coupons_len.cumsum())
        
        pv_coupons = coupons_len / 360  * swap_coupon / (1 + discount_rates * coupons_len.cumsum() / 360)
        pv_principal = 1 / (1 + discount_rates[-1] * coupons_len.cumsum()[-1] / 360)
        
        return np.sum(pv_coupons) + pv_principal
    
    def estimate_pillar_discount_rate(self, nflows, i0=0):
        """
        Estimate the nth rate discount for a nx1 market swap,
        assuming all other discount factors are known.
        """
        rate_err = lambda rate: 1e6 * (self.estimate_bond_price(rate, nflows) - 1) ** 2
        rate_result = minimize(rate_err, i0, tol=1e-20)
        return rate_result
    
    def fit_curve(self, i0=0):
        """
        """
        for ix, nflow in enumerate(self.number_flows):
            irate = self.estimate_pillar_discount_rate(nflow, i0)
            self.discount_rates[ix] = irate.x[0]
        self.fitted_curve = True
        self.rate_curve = interp1d(self.days_to_pillar, self.discount_rates)
