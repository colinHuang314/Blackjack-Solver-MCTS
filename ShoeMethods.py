from collections import Counter
import random

def generateShoe(cards_to_remove, num_decks):
    shoe = Counter({i: 4 * num_decks for i in range(2, 10)})
    shoe[10] = 16 * num_decks
    shoe[11] = 4 * num_decks
    #remove cards
    for card in cards_to_remove:
        shoe[card] -= 1
    
    return shoe

def countOfShoe(shoe): # true count
    running_count = 0
    num_cards = 0
    for card, count in shoe.items():
        if card <= 6:
            running_count -= count
        elif card >= 10:
            running_count += count
        num_cards += count
    
    return running_count * 52 / num_cards # true count
    
    #makes the count barely, so not the average of counts in this true count
    #make count and half to simlulate average of hands at this count?
def makeCountOfShoe(shoe, tc): #doesn't account for deck size shrinking but should be a big difference, also it airs on the safe side (higher count)
    '''
    get current running count and the running count you need and subtract to get how many cards to remove
    '''
    running_count = 0
    num_cards = 0
    for card, count in shoe.items():
        if card <= 6:
            running_count -= count #opposite since its cards in shoe not removed
        elif card >= 10:
            running_count += count
        num_cards += count
    
    # running_count_needed = round((tc + (0.5 if tc > 0 else -0.5)) * num_cards / 52) ############################################################################ 0.5, 0.4 round down?^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    running_count_needed = round((tc + 0.25) * round(num_cards / 52)) # 0.25 to be conservative, but take into account that you want to represent all hands at that count which could be anywhere from tc to (tc + 0.99) since you round down
    difference_in_running_count = running_count_needed - running_count
    # print("tc goal: " + str(tc))
    # print(f"running: {running_count}, needed: {running_count_needed}, diff: {difference_in_running_count}") 37
    ''' Try to make average representation of deck count'''
    low_cards = [2,3,4,5,6,7,8,9]
    high_cards = [10,10,10,10,11,7,8,9]
    warnings = 0
    while difference_in_running_count != 0:
        if difference_in_running_count > 0: #remove low cards
            if len(low_cards) == 0:
                low_cards = [2,3,4,5,6,7,8,9]
            card = random.choice(low_cards)    
            low_cards.remove(card)
            if shoe[card] == 0:
                warnings += 1
                continue
            else:
                shoe[card] -= 1
            if card <= 6:
                difference_in_running_count -= 1
        else: #remove high cards
            if len(high_cards) == 0:
                high_cards = [10,10,10,10,11,7,8,9]
            card = random.choice(high_cards)
            high_cards.remove(card)
            if shoe[card] == 0:
                warnings += 1
                continue
            else:
                shoe[card] -= 1
            if card >= 10:
                difference_in_running_count += 1
        if warnings >= 1000:
            assert ValueError("error: cant make count of deck due to not enough cards to remove")

#made by ChatGPT (modified by me)
def drawCardFromShoe(shoe, excludedCards = []):
    filtered = [(card, count) for card, count in shoe.items() if count > 0 and card not in excludedCards]

    cards, weights = zip(*filtered)
    drawn = random.choices(cards, weights=weights, k=1)[0]
    shoe[drawn] -= 1
    return drawn