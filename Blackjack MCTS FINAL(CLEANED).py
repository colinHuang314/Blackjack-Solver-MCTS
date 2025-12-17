# from collections import Counter
import random
import math
import time
from colorama import Fore
import locale

import GameState
# import DealerHand
import PlayerHand
import ShoeMethods

import OptionAnalyzer

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') # uses commas


# --- Input Game State ---
hand = [10,4] # Ex: [11, 11] for 2 aces
dealer_card = 3

count = 0 # put 0 for basic strategy

# --- Function ---
get_deviations = False # settings apply to deviations the same, but per count it tries in semi-binary search (expect it to take 5x longer)


# --- Can Change These ---
PRINT_EXTRAS = False # prints extra information
RUNTIME = 3 # run time for algorithm (choose any)
STOP_MAIN_BRANCH_WHEN_CONFIDENT = False # keeps running until it is confident enough (depends on setting on line 44) and ignores runtime
if STOP_MAIN_BRANCH_WHEN_CONFIDENT: RUNTIME = 86400
use_basic_strategy_in_rollouts = True # more accurate rollouts in splits, otherwise early searches will pollute split values 

# --- Game Rules (can change) --- 
SURRENDER = False # not simulated, is just -0.5 ev
NUM_DECKS = 4 # simulates with this amount of decks, does not account for changing deck size while playing. # could run out of cards in deck if too low and max_splits is high
MAX_SPLITS = 3 # 3 means splitting into 4 hands
H17 = True # dealer hits on soft 17
ACES_GET_1_CARD = True

# Recomended to use "low", "medium" or "tuned"
if STOP_MAIN_BRANCH_WHEN_CONFIDENT:
    setting = "low"
else: 
    setting = "none"

###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################

# --- Complicated parameters, not reccomended to change ---
C_PARAM = 3 # exploration constant for MCTS
MIN_SIMS = 15_000 # 10_000 - 30_000
BATCH_SIZE = 50_000 # starting batch size 60_000

EPSILON = 0.0002 # 0.0001-4 #distance between current and past ev to be considered not changed -- 0.001 for 5 mil itters, 0.003 for 1 mil
CONFIDENCE_SAMPLE_SIZE = 250 # 250 good
MIN_EXPANSIONS_FOR_SPLITS = 1_000_000 # 1 mil (safety, confidence doesn't 100% work for MCTS which just needs itters)
MIN_CONFIDENCE_FOR_STOPPING_BRANCH = 0.56 
UNLOCK_ACTION_RATIO = 4
PRE_COOK_ITTERS = 0 # 1 mil 

EXPLORE_MODE = False
ACCURATE_EV_MODE = False # simulates all actions equally

# --- Debug ---
TRAVERSE_TREE = False # traverse tree fter simulation in console


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################

def basicStrategy(state):
    if state.isTerminal():
        raise ValueError("State is terminal")
    
    curr_hand = state.player_hands[state.hand_index]
    key = state.getKey() 

    options = basicStrategyTable[key].split(",") #(will have "" at the end)
    for i in range(len(options)):
        if options[i] == "split" and not "split" in curr_hand.legal_actions: 
            continue
        if options[i] == "double" and not "double" in curr_hand.legal_actions:
            continue
        if options[i] == "surrender" and not "surrender" in curr_hand.legal_actions:
            continue
        else:
            return options[i]
        
def loadCachedMoves(file_path): #into dictionary
    moves = {}
    f = open(file_path, "r")
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip() #remove return char

        if ";" in line:
            key, options = line.split(";")
            moves[key] = options
    
    f.close()
    return moves
#########################################################################################################
class MCTSNode:
    def __init__(self, state, node_type = "player", action_taken=None, chance_action=None, untried_actions=None, parent=None):
        self.state = state
        self.node_type = node_type
        self.parent = parent
        self.children = []
        self.child_probabilities = [] #for chance
        self.chance_action = chance_action # for chance
        self.done_simulating = False # for chance -- when ev doesnt change for a while, stop simulating this branch
        self.visits = 0
        self.value = 0.0
        self.untried_actions = untried_actions if not (untried_actions == None) else state.legalActions().copy() #what actions would be legal if action was not terminal
        self.action_taken = action_taken #aka how did i get here?

    def updateUntriedActions(self):
        self.untried_actions = self.state.legalActions().copy()

    def isFullyExpanded(self): #reached all children
        return self.node_type == "chance" or len(self.untried_actions) == 0 #immediately fully expand chance nodes

    def best_choice(self): #for final move (called by root)
        return max(self.children, key=lambda c: (c.visits)) # or ev

    #3 worked for non-splits, struggled on some doubles
    def best_child(self, c_param):#which child to choose based on exploitation vs exploration formula (assume player state
        best_score = -float("inf")
        best_children = [] # if theres a tie

        #print(f"num of children: {len(self.children)}")
        for child in self.children:
            if child.visits == 0:
                score = float("inf")  # try unvisited nodes
            elif (ACCURATE_EV_MODE or EXPLORE_MODE) and self.action_taken == None: #first branch -- if explore mode, explore done simulating actions
                score = float("inf")
            elif child.done_simulating: # dont simulate
                score = -float("inf")

            else:
                exploit = child.value / child.visits
                explore = c_param * math.sqrt(math.log(self.visits) / child.visits)
                score = exploit + explore

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return random.choice(best_children)

    def simulateChanceNode(self): #called on chance node to return action + card drew
        return random.choices(self.children, weights=self.child_probabilities, k=1)[0]

    '''
    calling expand on a node assumes that this node is "player", is not fully expanded and is not terminal (has legal actions untried)
    we choose a random action to try out of untried_actions
    create chance node as child
    we create a player child node and update the state to reflect the result of each possible outcome of the action ex:"hit 4"
    '''
    def expand(self):
        action = self.untried_actions.pop()
        chance_node = MCTSNode(self.state, "chance", self.action_taken, action, [], self) #chance nodes dont need untried_actions
        self.children.append(chance_node)

        #expand chance node
        if action == "stand":
            player_node = MCTSNode(self.state, "player", "stand", None, [], chance_node)
            chance_node.children.append(player_node)
            chance_node.child_probabilities.append(1)

        elif action == "double":
            for card, count in self.state.shoe.items():
                player_node = MCTSNode(self.state, "player", "double " + str(card), None, [], chance_node)
                chance_node.children.append(player_node)
                chance_node.child_probabilities.append(count)

        elif action == "hit":
            for card, count in self.state.shoe.items():
                player_node = MCTSNode(self.state, "player", "hit " + str(card), None, ["hit", "stand"], chance_node)
                chance_node.children.append(player_node)
                chance_node.child_probabilities.append(count)
        
        elif action == "split": #split and one auto hit
            
            for card, count in self.state.shoe.items():
                actions = []
                if card == self.state.player_hands[self.state.hand_index].hand[0]:
                    if len(self.state.player_hands) < MAX_SPLITS: ############never false?
                        actions += ["split"]
                   
                actions += ["double"]
                actions += ["hit", "stand"]
                player_node = MCTSNode(self.state, "player", "split " + str(card), None, actions, chance_node)
                chance_node.children.append(player_node)
                chance_node.child_probabilities.append(count)


    def backpropagate(self, reward): #add value to parents and add 1 to times visited
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def rollout(self): #on either node type #sudo-random monte carlo simulation to get an idea of the value of position fast
        #state altered
        while not self.state.isTerminal():
            if use_basic_strategy_in_rollouts:
                action = basicStrategy(self.state)
            else:
                action = random.choice(self.state.legalActions())

            self.state.performAction(action)
        return self.state.getReward()
    
    def printBestChildren(self, separate_line = False):
        childList = self.getActionsAndEVs()
        for action, EV in childList:
            #while len(action) < 9: action += " " #padding
            print(Fore.LIGHTBLUE_EX + action + Fore.GREEN, end = "")
            print(f" EV: {EV}", end = ("    " if not separate_line else "\n"))
    
    def getActionsAndEVs(self):
        child_list = []
        for child in self.children:
            action = child.chance_action
            EV = round(child.value / child.visits, 4)
            child_list.append((action, EV))
        if SURRENDER:
            child_list.append(("surrender", -0.5))
        
        return sorted(child_list, key=lambda obj: (obj[1]), reverse=True)


    def __repr__(self):
        if self.node_type == "player":
            txt = "\tPlayer Node: \n"
            if self.visits == 0:
                txt += "Action: " + str(self.action_taken) + ", no visits\n"
            else: 
                txt += "Action: " + str(self.action_taken) + ", EV: " + str(round(float(self.value / self.visits),4)) + ", Visits: " + locale.format_string("%d", self.visits, grouping=True)
                txt += ", Legal actions: " + str(self.state.legalActions()) + "\n"
                for i, child in enumerate(self.children):
                    print(f"Child {i}: {child.chance_action}, EV: {round(child.value/child.visits,4)}, Visits: {child.visits}")
        else:
            txt = "\tChance Node: \n"
            if self.visits == 0:
                txt += "Action: " + str(self.action_taken) + ", no visits\n"
            else: txt += "Action: " + str(self.action_taken) + ", EV: " + str(round(float(self.value / self.visits),4)) + ", Visits: " + locale.format_string("%d", self.visits, grouping=True) + "\n"
            
            txt += "Possible outcomes: \n"
            for i, child in enumerate(self.children):
                txt += f"Child {i}: {child.action_taken}, EV: {round(float(child.value / child.visits),4)}, Visits: {child.visits}, Weight: {self.child_probabilities[i]}\n"
                
        return txt


def findDeviations(starting_state, itterations = 100_000, runtime = 0):
    sim_data = []
    #count of 0
    starting_shoe = starting_state.shoe.copy()
    print(Fore.YELLOW + "Calculating Baseline (count 0)")
    start = time.time()
    actionsAndEVs, conf = bestMoveMCTS(starting_state, RUNTIME)
    sim_data.append((0, round(time.time() - start), round(conf,4)))
    baseline = actionsAndEVs[0][0]
    ev1 = actionsAndEVs[0][1]
    ev2 = actionsAndEVs[1][1]
    baseline_difference = ev1 - ev2
    step = 4 # try every 4 true counts
    direction = 1 # guess
    action = baseline
    prev_difference = baseline_difference
    deviation = None
    while True:
        true_count = direction * step
        
        starting_state.shoe = starting_shoe.copy()
        ShoeMethods.makeCountOfShoe(starting_state.shoe, true_count)
        print(Fore.YELLOW + "\n\t\tTrying count of " + str(true_count) + "(.3)")
        start = time.time()
        actionsAndEVs, conf = bestMoveMCTS(starting_state, RUNTIME)
        sim_data.append((true_count, round(time.time() - start), round(conf,4)))

        action = actionsAndEVs[0][0]
        ev1 = actionsAndEVs[0][1]
        ev2 = actionsAndEVs[1][1]        
        difference = ev1 - ev2
        if action != baseline:
            deviation = action
            lo = min(true_count, true_count - direction*4) 
            hi = max(true_count, true_count - direction*4)

            if direction == 1: #avoid unnessesarry search
                lo +=1
            else: hi -=1

            while lo < hi:

                true_count = ((lo + hi) // 2) if direction == 1 else ((lo + hi + 1) // 2) #round towards 0

                starting_state.shoe = starting_shoe.copy()
                ShoeMethods.makeCountOfShoe(starting_state.shoe, true_count)
                print(Fore.YELLOW + "Trying count of " + str(true_count))
                start = time.time()
                actionsAndEVs, conf = bestMoveMCTS(starting_state, RUNTIME)
                sim_data.append((true_count, round(time.time() - start), round(conf,4)))
                action = actionsAndEVs[0][0]
                if direction == 1:
                    if action == baseline:
                        lo = true_count + 1
                    else:
                        hi = true_count
                else:
                    if action == baseline:
                        hi = true_count - 1
                    else:
                        lo = true_count
            return baseline, deviation, lo, sim_data
        
        #if action not changed:
        if difference > prev_difference: #wrong way
            if abs(step) > 4:
                print("SOMETHING IS WRONG, got farther from answer")
            direction *= -1
        else: #try way first
            step += 4
        prev_difference = difference


def getChild(node: "MCTSNode", action):
    for child in node.children:
        if child.chance_action == action:
            return child
    raise ValueError

def changeSettings(level):
    global MIN_SIMS, EPSILON, BATCH_SIZE, CONFIDENCE_SAMPLE_SIZE, MIN_EXPANSIONS_FOR_SPLITS, MIN_CONFIDENCE_FOR_STOPPING_BRANCH, PRE_COOK_ITTERS
    if level == "fastest":
        MIN_SIMS = 500
        EPSILON = 0.04
        BATCH_SIZE = 5_000 
        CONFIDENCE_SAMPLE_SIZE = 6
        PRE_COOK_ITTERS = 0
        MIN_EXPANSIONS_FOR_SPLITS = 75_000

    if level == "fast":
        MIN_SIMS = 1_000
        EPSILON = 0.02
        BATCH_SIZE = 5_000 
        CONFIDENCE_SAMPLE_SIZE = 15
        PRE_COOK_ITTERS = 0
        MIN_EXPANSIONS_FOR_SPLITS = 120_000

    if level == "low":
        MIN_SIMS = 1_000
        EPSILON = 0.01
        BATCH_SIZE = 10_000 
        CONFIDENCE_SAMPLE_SIZE = 25
        PRE_COOK_ITTERS = 0
        MIN_EXPANSIONS_FOR_SPLITS = 150_000

    if level == "tuned":
        MIN_SIMS = 15_000
        EPSILON = 0.0002 #-- depends on batch size
        BATCH_SIZE = 40_000 # > 2_000
        CONFIDENCE_SAMPLE_SIZE = 250 # 
        MIN_EXPANSIONS_FOR_SPLITS = 1_000_000
        MIN_CONFIDENCE_FOR_STOPPING_BRANCH = 0.56 #depends on confidence sample size
        PRE_COOK_ITTERS = 1_000_000

    if level == "medium":
        MIN_SIMS = 5_000
        EPSILON = 0.001
        BATCH_SIZE = 50_000 
        CONFIDENCE_SAMPLE_SIZE = 40 
        PRE_COOK_ITTERS = 300_000
        MIN_EXPANSIONS_FOR_SPLITS = 600_000

    if level == "high":
        MIN_SIMS = 10_000 
        EPSILON = 0.0001
        BATCH_SIZE = 60_000 
        CONFIDENCE_SAMPLE_SIZE = 250
        PRE_COOK_ITTERS = 1_000_000

    if level == "best":
        MIN_SIMS = 10_000 
        EPSILON = 0.00001
        BATCH_SIZE = 10_000_000 
        CONFIDENCE_SAMPLE_SIZE = 250
        PRE_COOK_ITTERS = 2_000_000


def bestMoveMCTS(root_state, runtime, itterations = 30_000): # min itterations
    global EXPLORE_MODE
    EXPLORE_MODE = False

    print_out = False
    root_node = MCTSNode(root_state)
    start = time.time()
    optionAnalyzer = OptionAnalyzer.OptionAnalyzer(CONFIDENCE_SAMPLE_SIZE)
    first_action_taken = None
    saved_first_action = False
    itters = 0
    update_number = 1
    last_actions_and_EVs = None
    done_simulating = False
    expansions = 0
    confidence = -1
    print(Fore.RESET + "-------------------------------------------------------------------------------------------")
    print(root_state.player_hands[0])
    print(Fore.LIGHTBLUE_EX + "Dealer " + str(root_state.dealer_card) + Fore.RESET)
    print("\nUpdating every " + locale.format_string("%d", BATCH_SIZE, grouping=True) + " itterations. EPSILON: " + (str(EPSILON) if STOP_MAIN_BRANCH_WHEN_CONFIDENT else "N/A"))
    print(Fore.MAGENTA + "explore param: " + str(C_PARAM) + ", MAX SPLITS: " + str(MAX_SPLITS) + Fore.RESET)
    if PRE_COOK_ITTERS: 
        EXPLORE_MODE = True
        print("Pre cooking for " + locale.format_string("%d", PRE_COOK_ITTERS, grouping=True) + " itters, cooking with" + ("" if EXPLORE_MODE else "out") + " explore mode (will be left on after done)")
    while ((time.time() - start < runtime or itters < itterations) and not done_simulating):
        saved_first_action = False

        if itters - PRE_COOK_ITTERS > BATCH_SIZE * update_number: # print out
            print(Fore.YELLOW + "Update at itter " + locale.format_string("%d", round(update_number * BATCH_SIZE), grouping=True) + " ", end=" (After precook)")
            update_number += 1

            root_node.printBestChildren()
            print("\t" + Fore.RESET + "Time: " + Fore.YELLOW + str(round(time.time() - start, 3)), end="")
            print()

            if STOP_MAIN_BRANCH_WHEN_CONFIDENT:

                # stop simulating options that are sure
                actions_and_EVs = root_node.getActionsAndEVs() # list of (action, ev)

                actions_to_compare = [actions_and_EVs[0][0]] #top action -- if done, compare to next not done, if not done, compare to next
                if actions_and_EVs[0][0] != "surrender" and not getChild(root_node, actions_and_EVs[0][0]).done_simulating:
                    actions_to_compare.append(actions_and_EVs[1][0])
                else:
                    for i in range(len(actions_and_EVs)): #next best thats not node simulating appended 
                        if i == 0: continue
                        if actions_and_EVs[i][0] != "surrender" and not getChild(root_node, actions_and_EVs[i][0]).done_simulating:
                            actions_to_compare.append(actions_and_EVs[i][0])
                            break

                confidence = optionAnalyzer.confidence(actions_to_compare[0], actions_to_compare[1])
                if PRINT_EXTRAS: print(Fore.RED + "Confidence that " + actions_to_compare[0] + " is better than " + actions_to_compare[1] + ": " + str(round(float(confidence), 4)))
                if PRINT_EXTRAS: print("expansions: " + locale.format_string("%d", expansions, grouping=True))

                if (confidence > 0.99):
                    if all([child.visits > MIN_SIMS or child.done_simulating for child in root_node.children] ): #all nodes get sufficient sims
                        if (not "split" in root_state.legalActions() or (expansions > MIN_EXPANSIONS_FOR_SPLITS) or (root_state.player_hands[0] == [11,11] and ACES_GET_1_CARD)) : #expansions > MIN_EXPANSIONS and 
                            done_simulating = True
                            if print_out: print("confidence is high, done simulating")
                    
                    if not done_simulating:
                        if EXPLORE_MODE == False:
                            EXPLORE_MODE = True
                            for child in root_node.children:
                                child.done_simulating = False
                            if PRINT_EXTRAS: print("not hit min sim req of 10,000 for all actions or enough expansions for splitting, activating explore mode, unlocking all actions")
                else:
                    if EXPLORE_MODE == True:
                        EXPLORE_MODE = False
                        if PRINT_EXTRAS: print("confidence dropped, disabling explore mode")


                #CHECK IF WE CAN STOP SIMULATING A MOVE
                if EPSILON:#'''enabled''' only is effective if nodes being visited
                    if last_actions_and_EVs != None: # first check has something to compare to, 
                        for i, pair in enumerate(actions_and_EVs):
                            if pair[0] == "surrender": #dont try to stop surrender
                                continue
 
                            if pair[0] == last_actions_and_EVs[i][0]: #ranking didnt change

                                #EPSILON + confidence * 10 * 0.0001
                                #pow((1+confidence), 5) * 0.00005
                                if abs(pair[1] - last_actions_and_EVs[i][1]) <= EPSILON: 
                                    child = getChild(root_node, pair[0])
                                    if not child.done_simulating:
                                        if confidence > MIN_CONFIDENCE_FOR_STOPPING_BRANCH: 
                                            if (child.chance_action != "split" or (expansions > MIN_EXPANSIONS_FOR_SPLITS) or (root_state.player_hands[0] == [11,11] and ACES_GET_1_CARD)):#dont stop split if not enough sims
                                                child.done_simulating = True
                                                if PRINT_EXTRAS: print(Fore.LIGHTMAGENTA_EX  + pair[0] + " has not changed ev much, stopping their simulations, Visits: " + locale.format_string("%d", child.visits, grouping=True) + ", confidence: " + str(confidence)) # or their ev are very close (within 3 decimals)
                                            else:
                                                if PRINT_EXTRAS: print(f"{Fore.RESET}not enough split sims ({locale.format_string("%d", expansions, grouping=True)}/{locale.format_string("%d", MIN_EXPANSIONS_FOR_SPLITS, grouping=True)}) to stop simulating split")
                                            if all(child.done_simulating for child in root_node.children): #if all actions done simulating, stop simulations
                                                if abs(actions_and_EVs[0][1] - actions_and_EVs[1][1]) > 6 * EPSILON: #treating e as std, 3 std for each action = 99.7%
                                                    if print_out: print("ALL EV STABLE AND EVS ARE FAR ENOUGH APART, EFFECTIVE 99.9% confidence")
                                                else:
                                                    if print_out: print("ALL EV STABLE BUT ACTIONS ARE VERY CLOSE, ONLY SOMEWHAT CONFIDENT")
                                                done_simulating = True
                                            break # only stop simulating the best node(most visited) that isnt done simulating (worst nodes wont be simulated and wont change ev much
                                        else:
                                            c1 = getChild(root_node, actions_to_compare[0])
                                            c2 = getChild(root_node, actions_to_compare[1])
                                            if max(c1.visits, c2.visits) / min(c1.visits,c2.visits) >= UNLOCK_ACTION_RATIO:
                                                for child in [c1, c2]:
                                                    if child.done_simulating:
                                                        child.done_simulating = False
                                                        print("need more simulations, unlocking: " + child.chance_action)
                                            else:
                                                if PRINT_EXTRAS: print("confidence not high enough(" +str(MIN_CONFIDENCE_FOR_STOPPING_BRANCH)+ ") to stop simulating " + pair[0] + ", confidence: " + str(confidence) + ", visits: " + str(child.visits))
                                    else:
                                        continue
                                    
                            break
                    last_actions_and_EVs = actions_and_EVs
                        
                
        itters += 1
        action_path = []
        curr_node = root_node

        '''
        # --- MCTS ---
        '''
        #selection
        while not curr_node.state.isTerminal() and curr_node.isFullyExpanded():
            try:
                curr_node = curr_node.best_child(C_PARAM)
                if not saved_first_action:
                    first_action_taken = curr_node.chance_action
                    saved_first_action = True
            except:
                print(curr_node.state)
                print(curr_node.state.isTerminal())
                print(curr_node.untried_actions)
                raise IndexError

            curr_node = curr_node.simulateChanceNode() #get next player node
            action_path.append(curr_node.action_taken)
            
            terminated = curr_node.state.performAction(curr_node.action_taken)
            if terminated and not curr_node.state.isTerminal():
                curr_node.updateUntriedActions()

        #expansion
        if not curr_node.state.isTerminal():
            curr_node.expand() #expanding player node
            expansions += 1
      
        #simulation
        value = curr_node.rollout() # only 1

        #add to option analyzer for confidence estimation
        if first_action_taken != None:
            optionAnalyzer.addResult(first_action_taken, value)

        #backpropagation
        curr_node.backpropagate(value)
        #print("After backpropagation: " + str(curr_node))

        #revert state
        curr_node.state.reverseActions()

    # print("\n\nroot: " + str(root_node))
    
    print(Fore.RED + "-------------------------------------------------------------------------------------------")
    
    print("Final Calculation: " + Fore.MAGENTA + "(explore param: " + str(C_PARAM) + ", MAX SPLITS: " + str(MAX_SPLITS) + ")" + Fore.RESET)

    actions_and_EVs = root_node.getActionsAndEVs() # list of (action, ev)
    
    if PRINT_EXTRAS:
        print(optionAnalyzer.option_data)
        #confidence = optionAnalyzer.confidence(actions_and_EVs[0][0], actions_and_EVs[1][0])
        if confidence > 0.99:
            confidence = 0.99
        if confidence == -1:
            print(Fore.RED + "Confidence not tracked")
        else:
            print(Fore.RED + "Confidence: " + str(round(float(confidence),4)) + " for " + str(actions_and_EVs[0][0]), end="")
        print(Fore.RESET + "(EV for less visited nodes is less accurate)")
        print("expansions: " + str(expansions))
    root_node.printBestChildren(separate_line = True)
    print()
    print(f"Time took: {(time.time() - start):.3f}, Itterations: {locale.format_string("%d", itters, grouping=True)}")
    
    if PRINT_EXTRAS: print(Fore.RESET + str(curr_node.state.shoe))
    print(Fore.RESET, end="")
    
    if TRAVERSE_TREE:
        inp = input(Fore.RED + "Press enter to begin tree traversal (q to quit)" + Fore.RESET)
        curr_node = root_node
        while inp != "q":
            print(curr_node)
            inp = input("which number child to explore? (r to reset, q to quit)")
            if inp == "q": break
            if inp == "r":
                curr_node = root_node
                continue
            curr_node = curr_node.children[int(inp)]


    return root_node.getActionsAndEVs(), confidence
    
#########################################################################################################
#########################################################################################################
def getDeviationMCTS(starting_state,  itterations = 100_000, runtime = 0):
    start = time.time()
    action_at_0, deviation, count_to_deviate, sim_dataList = findDeviations(starting_state, itterations, runtime)
    time_taken = time.time() - start
    print()
    print(starting_state.player_hands[0])
    print("vs dealer " + str(starting_state.dealer_card))
    print(Fore.RED)
    print(f"Deviate by doing: {deviation}  instead of: {action_at_0}  at true count {count_to_deviate}")
    for data in sim_dataList:
        print(Fore.LIGHTBLUE_EX, end="")
        print(f"Time taken for count {data[0]}: {data[1]}, confidence: {data[2]}")
    print(Fore.YELLOW + "Total time taken: " + str(round(time_taken)))
#########################################################################################################
#########################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
changeSettings(setting)
print("settings changed to " + setting) 

if not use_basic_strategy_in_rollouts:
    print("using random rollouts")

basicStrategyTable = loadCachedMoves("Projects\\blackjack\\basicStrategyCache.txt") #used in basicStrategy()
print(Fore.RESET, end="")

starting_legal_actions = []
if hand[0] == hand[1]:
    starting_legal_actions += ["split"]
starting_legal_actions += ["hit", "double", "stand"]

starting_player_hand = PlayerHand.PlayerHand(hand, 1, starting_legal_actions)
#starting_player_hand2 = PlayerHand([7,8], 1, ["hit","double","stand"])

starting_player_hands = [starting_player_hand]#, starting_player_hand2]

starting_cards_to_remove = [dealer_card] #
for hand in starting_player_hands:
    starting_cards_to_remove += hand.hand
starting_shoe = ShoeMethods.generateShoe(starting_cards_to_remove, NUM_DECKS)

starting_state = GameState.GameState(starting_player_hands, dealer_card, starting_shoe, H17, MAX_SPLITS, ACES_GET_1_CARD)

ShoeMethods.makeCountOfShoe(starting_state.shoe, count)
print("\nsimulating at a count of " + str(count) + ".")

if not get_deviations:
    actions_and_ev, _= bestMoveMCTS(starting_state, RUNTIME)
    if PRINT_EXTRAS:
        if actions_and_ev[0][0] == basicStrategy(starting_state):
            print("matches basic strategy")
        else:
            print(Fore.YELLOW + "not basic strategynot basic strategynot basic strategynot basic strategynot basic strategynot basic strategynot basic strategynot basic strategynot basic strategy")
else:
    getDeviationMCTS(starting_state)#, itterations=1_000_000, runtime=0) # about 4 times as long + if a count is very close could be more 12 times?


print(Fore.RESET)
if PRINT_EXTRAS: print("dev notes for future updates:  could disregard first n itters by keeping running sum of totals of first action nodes, fix tc calc for make deck count, 16 vs 9, more rules, ")