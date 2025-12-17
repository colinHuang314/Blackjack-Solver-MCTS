
from scipy.stats import norm
import math


'''
Help from google and chatgpt
'''
class OptionAnalyzer:
    def __init__(self, sample_size = 250):
        self.option_data = {}
        self.addResult("surrender", -0.5)
        self.option_data["surrender"]["n"] = -1
        self.sample_size = sample_size

    def addResult(self, option, result):
        if not option in self.option_data:
            self.option_data[option] = {
                "n": 0.0,
                "sum": 0.0,
                "sum_of_squares": 0.0
            }
        data = self.option_data[option]
        data["n"] += 1
        data["sum"] += result
        data["sum_of_squares"] += result * result

    def mean(self, option):
        if option == "surrender": return -0.5
        data = self.option_data[option]
        return data["sum"] / data["n"] if data["n"] > 0 else -9999999999
    
    def stddev(self, option):
        if option == "surrender": return 0
        data = self.option_data[option]
        if data["n"] < 2:
            return 0.0
        mean_sq = data["sum"] * data["sum"] / data["n"] 
        variance = (data["sum_of_squares"] - mean_sq) / (data["n"] - 1)
        return math.sqrt(variance)
    
    #faster  300,000 itters  2000 ss -62% confidence  1000ss -70%  500ss 76%  250ss -86%
    def confidence(self, option_a, option_b): # 250 #sample size is an estimated number of rollouts to account for rollout varience (dealer making a hand)
        # print("option 1: " + option_a)
        # print("option 2: " + option_b)
            
        nA = self.option_data[option_a]["n"]
        nB = self.option_data[option_b]["n"]

        if nA == -1: nA = nB*3 #surrender has no n, use other n instead
        if nB == -1: nB = nA*3 # *3 since surrender is exactly -0.5 (not sure the correct math but *3 seems ok)
        
        n = (nA + nB) // 2

        stdA = self.stddev(option_a)
        stdB = self.stddev(option_b)
        meanA = self.mean(option_a)
        meanB = self.mean(option_b)

        pooled_se = math.sqrt((stdA**2)/(n/self.sample_size) + (stdB**2)/(n/self.sample_size))  #accounts for selective sampling in MCTS
        z_score = (meanA - meanB) / pooled_se
        confidence = norm.cdf(z_score)  # probability that A > B

        return confidence
