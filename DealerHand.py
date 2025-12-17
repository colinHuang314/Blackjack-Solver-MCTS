class DealerHand:
    def __init__(self, dealer_card, H17):
        self.hand = [dealer_card]
        self.hand_total = dealer_card
        self.h17 = H17
    
    def addCard(self, card):
        self.hand.append(card)
        total = sum(self.hand)
        aces = self.hand.count(11)
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        
        if total > 16:
            if not(aces and self.h17 and total == 17):
                return total
        
        return 0