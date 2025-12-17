import DealerHand
import PlayerHand
import ShoeMethods

class GameState: #used to reach all states so no copying
    def __init__(self, player_hands, dealer_card, shoe, H17, MAX_SPLITS, ACES_GET_1_CARD):
        self.player_hands = player_hands
        self.hand_index = 0
        self.dealer_card = dealer_card
        self.shoe = shoe
        self.action_stack = []

        self.h17 = H17
        self.max_splits = MAX_SPLITS
        self.aces_get_1_card = ACES_GET_1_CARD

    def getKey(self):
        curr_hand = self.player_hands[self.hand_index]
        dealer_card = self.dealer_card
        if "split" in curr_hand.legal_actions: 
            if curr_hand.is_soft:
                return "pair " + "11" + " vs " + str(dealer_card)
            return "pair " + str(curr_hand.hand_total//2) + " vs " + str(dealer_card)
        
        return ("soft " if curr_hand.is_soft else "hard ") + str(curr_hand.hand_total) + " vs " + str(dealer_card)
    
    def __repr__(self):
        txt = "Hands: \n"
        for i, hand in enumerate(self.player_hands):
            txt += "Current:\t" if i == self.hand_index else "\t\t"
            txt +=  str(i) + ". " + str(hand) + "\n"
        txt += "Dealer Card: " + str(self.dealer_card) + "\n"
        txt += "Shoe: " + str(self.shoe) + "\n"
        txt += "Actions performed: " + str(self.action_stack) + "\n"
        return txt

    def legalActions(self):
        return self.player_hands[self.hand_index].legal_actions
    
    def performActions(self, actions):
        while actions:
            self.performAction(actions.pop(0))

    def reverseActions(self):
        #print("Actions to be reversed: " + str(self.action_stack)) ####ps
        #print(self.player_hands)
        while self.action_stack:
            self.reverseAction(self.action_stack.pop())

    def performAction(self, action):
        action_list = action.split(" ")
        if len(action_list) == 2:
            action_string, card = action_list
            card = int(card)
        else:
            action_string = action
            card = None

        if action_string == "stand":
            self.player_hands[self.hand_index].stand()
        elif action_string == "hit":
            if card == None: card = ShoeMethods.drawCardFromShoe(self.shoe)
            else: self.shoe[card] -= 1
            self.player_hands[self.hand_index].hit(card)
        elif action_string == "double":
            if card == None: card = ShoeMethods.drawCardFromShoe(self.shoe)
            else: self.shoe[card] -= 1
            self.player_hands[self.hand_index].double(card)
        # elif action_string == "surrender":
        #     self.player_hands[self.hand_index].surrender()
        elif action_string == "split":
            '''
            Create 2 new hand objects
            delete other hand
            hand_index is the same
            shoe changed for hit
            '''
            if card == None: card = ShoeMethods.drawCardFromShoe(self.shoe)
            else: self.shoe[card] -= 1

            old_hand = self.player_hands[self.hand_index].hand
            actions = []
            if old_hand[0] == card:
                if len(self.player_hands) < self.max_splits: #raraly occer
                    actions += ["split"]
            #if DAS: actions += ["double"]
            actions += ["hit", "double", "stand"]

            new_hand1 = PlayerHand.PlayerHand([old_hand[0]], 1, actions)
            new_hand2 = PlayerHand.PlayerHand([old_hand[1]], 1, ["hit", "double", "stand"])

            self.player_hands.pop(self.hand_index)
            self.player_hands.insert(self.hand_index, new_hand1)
            self.player_hands.insert(self.hand_index + 1, new_hand2)

            new_hand1.hit(card) #hit hand1

            if self.aces_get_1_card and new_hand1.hand[0] == 11: ###
                new_hand1.is_terminal = True
         

        else:
            raise ValueError("unknown action")
        
        self.action_stack.append(action) #remember what actions led to this state to undo it later
        
        #if current hand is terminal, index++
        if self.player_hands[self.hand_index].is_terminal:
            self.action_stack.append("+1")
            self.hand_index += 1

            if self.hand_index < len(self.player_hands) and len(self.player_hands[self.hand_index].hand) == 1: #has been split and need to auto draw for second hand
                card = ShoeMethods.drawCardFromShoe(self.shoe)
                curr_hand = self.player_hands[self.hand_index]
                curr_hand.hit(card)

                if self.aces_get_1_card and curr_hand.hand[0] == 11: #second hand of split aces
                    curr_hand.is_terminal = True
                    self.action_stack.append("hit " + str(card))
                    self.action_stack.append("+1")
                    self.hand_index += 1

                elif curr_hand.hand[0] == card:
                    if len(self.player_hands) < self.max_splits:
                        curr_hand.legal_actions = ["split"] + curr_hand.legal_actions #######mostly happen here, 

                    self.action_stack.append("hit " + str(card))
                else:
                    self.action_stack.append("hit " + str(card))


            return True #hand been terminated


    def reverseAction(self, action): #only called in reverseActions, actoins all revered at once
        if action == "+1":
            self.hand_index -= 1
            return
                
        action_list = action.split(" ")
        if len(action_list) == 2:
            action_string, card = action_list
            card = int(card)
        else:
            action_string = action

        if action_string == "stand":
            self.player_hands[self.hand_index].reverseStand()
        elif action_string == "hit":
            card = self.player_hands[self.hand_index].reverseHit()
            self.shoe[card] += 1
        elif action_string == "double":
            card = self.player_hands[self.hand_index].reverseDouble()
            self.shoe[card] += 1
        # elif action_string == "surrender":
        #     self.player_hands[self.hand_index].reverseSurrender()
        elif action_string == "split":
            '''
            remove index and index + 1
            create new hand with first card form each at index
            index is same, shoe is differnet
            '''
            card = self.player_hands[self.hand_index].reverseHit()
            self.shoe[card] += 1

            card1 = self.player_hands[self.hand_index].hand[0]
            card2 = self.player_hands[self.hand_index + 1].hand[0]
            actions = self.player_hands[self.hand_index].original_legal_actions
            if card1 == card2 and not "split" in actions:
                actions = ["split"] + actions
            new_hand = PlayerHand.PlayerHand([card1, card2], 1, actions)

            self.player_hands.pop(self.hand_index + 1)
            self.player_hands.pop(self.hand_index)
            self.player_hands.insert(self.hand_index, new_hand)
        else:
            raise ValueError("unknown action")
    
    def isTerminal(self):
        return all(hand.is_terminal for hand in self.player_hands)
    
    # def getReward(self):
    #     return sum((hand.hand_total if hand.hand_total <= 21 else 0) for hand in self.player_hands)
    def getReward(self): #lose/dealer draws (if dealer has Ace up card, assume no blackjack) (dont draw to blackjack)
        cardsDrawnInRollout = []
        
        # if self.player_hands[0].hand_total == 0: #surrender
        #     return -0.5 * self.player_hands[0].bet
        if len(self.player_hands) == 1:
            if len(self.player_hands) == 1 and len(self.player_hands[0].hand) == 2 and self.player_hands[0].hand_total == 21:#blackjack
                return 1.5
            

        profit = 0.0
        if all([hand.hand_total > 21 for hand in self.player_hands]): #all bust, dealer auto wins
            for hand in self.player_hands:
                profit -= hand.bet
            return profit

        # Dealer Turn
        dealer_hand = DealerHand.DealerHand(self.dealer_card, self.h17)
        exclude = 0
        if self.dealer_card == 11:
            exclude = 10
        elif self.dealer_card == 10: 
            exclude = 11

        card = ShoeMethods.drawCardFromShoe(self.shoe, [exclude]) #flips--dont have blackjack
        cardsDrawnInRollout.append(card)
        dealer_total = dealer_hand.addCard(card)

        while not dealer_total:            
            card = ShoeMethods.drawCardFromShoe(self.shoe) 
            cardsDrawnInRollout.append(card)
            dealer_total = dealer_hand.addCard(card)
        
        #print("Dealer total: " + str(dealer_total))

        for hand in self.player_hands: #check which hands win 
            player_total = hand.hand_total
            if player_total > 21:
                profit -= hand.bet
            elif dealer_total > 21:
                profit += hand.bet
            elif player_total > dealer_total:
                profit += hand.bet
            elif player_total < dealer_total:
                profit -= hand.bet
            #else push    

        # print(f"Hand: {self.player_hands[0]}")
        # print(f"dealer total: {dealer_total}")
        # print(f"profit: {profit}")
        for card in cardsDrawnInRollout: #add back cards
            self.shoe[card] += 1
        return profit