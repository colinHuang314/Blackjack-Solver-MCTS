from colorama import Fore

class PlayerHand:
    def __init__(self, hand, bet, legal_actions, is_terminal=False):
        self.hand = hand # might not need
        self.bet = bet
        self.legal_actions = legal_actions
        self.is_terminal = is_terminal
        self.hand_total, self.is_soft = self.handTotal()
        #self.is_surrendered = False
        self.original_legal_actions = self.legal_actions

    def handTotal(self):
        total = sum(self.hand)
        aces = self.hand.count(11)
        while total > 21 and aces > 0: #Down grade aces to benefit player
            total -= 10
            aces -= 1

        return total, aces #soft = if an ace is being used as 11
    
    def addCard(self, card):
        self.hand.append(card)
        self.hand_total, self.is_soft = self.handTotal()
        if self.hand_total > 21:
            self.is_terminal = True
        elif len(self.hand) > 2:
            self.legal_actions = ["hit", "stand"]

    def removeCard(self):  
        card = self.hand.pop()      
        self.hand_total, self.is_soft = self.handTotal()
        self.is_terminal = False
        if len(self.hand) <= 2:
            self.legal_actions = self.original_legal_actions
        return card
        
    #ACTIONS
    def stand(self):
        self.is_terminal = True
    def hit(self, card): #used to deal for splits
        self.addCard(card)
    def double(self, card):
        self.addCard(card)
        self.bet *= 2
        self.is_terminal = True
    # def surrender(self):
    #     self.is_surrendered = True
    #     self.is_terminal = True
    #for split creates new hand

    # REVERSALS
    def reverseStand(self):
        self.is_terminal = False
    def reverseHit(self):
        return self.removeCard()
    def reverseDouble(self):
        self.bet /= 2
        self.is_terminal = False
        return self.removeCard()
    # def reverseSurrender(self):
    #     self.is_surrendered = False
    #     self.is_terminal = False
    

    def __repr__(self):
        txt = "Hand: " + Fore.LIGHTBLUE_EX + str(self.hand) + Fore.RESET
        if "split" in self.legal_actions:
            txt += " (Pair of " + str(self.hand[0]) + "s)\n"
        else:
            txt += (" (Soft " if self.is_soft else " (Hard ") + str(self.hand_total) + ")\t"
        #txt += "Bet: " + str(self.bet) + "\n"
        txt += "TERMINAL" if self.is_terminal else ("Legal Actions: " + str(self.legal_actions))
        return txt