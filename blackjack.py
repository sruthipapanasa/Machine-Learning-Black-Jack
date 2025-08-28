import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import csv
import random
from dicts import hard_hands
from dicts import sph_dict



print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# column order in CSV file
column_names = ['card1', 'card2', 'card3', 'card4', 'card5', 'cardsum', 'dcard1', 'winloss']
class_names = ['Loss', 'Win']
feature_names = column_names[:-1]
label_name = column_names[-1]
print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

blackjackFile = "blackjack.csv"
modified_blackjack = 'modified_blackjack.csv'

with open(blackjackFile, 'r') as csvfile1:
    with open(modified_blackjack, 'w') as csvfile:
        pointreader = csv.reader(csvfile1)
        for row in pointreader:
            if row[15] == 'Win':
                row[15] = 1
            else:
                row[15] = 0
            temp = row[2:9]
            temp.append(row[15])
            row = temp
            pointwriter = csv.writer(csvfile)
            pointwriter.writerow(row)
            

#arrays
batch_size = 50
train_dataset = tf.data.experimental.make_csv_dataset(
    modified_blackjack,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)
features, labels = next(iter(train_dataset))
#print(features)

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels
train_dataset = train_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))
#print(features[:5])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(5,7)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(2)
])

model = keras.models.load_model('./blackjackmodel')



#actual game code

#setting up card
class Card:
    def __init__(self, suit, rank, value=0):
        self.suit = suit
        self.rank = rank
        self.value = value

    def show(self):
        print("{} of {} ({})".format(self.rank, self.suit, self.value))


class Deck:
    def __init__(self):
       self.cards = []
       
    def build(self):
        suits = ['Clubs', 'Spades', 'Diamonds', 'Hearts']
        ranks = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', 'Ten', 'Ten', 'Ten', 'Ten']
        values = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        index = 0
        for s in suits:
            for r in ranks:
                self.cards.append(Card(s,r, values[index%len(values)]))
                index += 1

    def shuffle(self):
        for i in range(len(self.cards)-1, 0, -1):
            r = random.randint(0, i)
            self.cards[i], self.cards[r] = self.cards[r], self.cards[i]
            
    def show(self):
        for c in self.cards:
            c.show()



#match parameters given to card in deck and returns card
def card_updater(deck, card_rank, card_suit):
    for card in deck.cards:
        if card.rank == card_rank:
            if card.suit == card_suit:
                return card

#hi-lo counter used in blackjack games
def counter(played_cards, count = 0):
    for card in played_cards.cards:
        if 2 <= card.value <= 6:
            count +=1
        elif 7 <= card.value <= 9:
            pass
        else:
            count -= 1
    return count

def model_prediction(model, player_hand, dealer_hand):
    predict_dataset = [0,0,0,0,0,0,0]
    for i in range(len(player_hand.cards)):
        predict_dataset[i] = player_hand.cards[i].value
    sum = 0
    for card in player_hand.cards:
        sum += card.value

    predict_dataset[5] = sum
    predict_dataset[6] = dealer_hand.cards[0].value
    predict_dataset = tf.convert_to_tensor([predict_dataset])

    predictions = model(predict_dataset, training=False)

    for i, logits in enumerate(predictions):
        class_idx = tf.argmax(logits).numpy()
        p = tf.nn.softmax(logits)[class_idx]
        name = class_names[class_idx]
        print("Outcome prediction if you end game now: {} ({:4.1f}%)".format(name, 100*p))

def prediction (count, player_hand, dealer_hand):
    #print sum of cards
    print("\n")
    sum = 0
    for card in player_hand.cards:
        sum += card.value
    print("The sum of your cards are:", sum)

    #uses model_prediction to display the win/loss rate at current hand
    model_prediction(model, player_hand, dealer_hand)

    #prints hi-lo count
    print("The current hi-lo count is:", str(count))

    #blackjack or not
    if sum > 21:
        print("You have busted. Better luck next time.")
    elif sum == 21:
        print("Blackjack! Congratulations.")
    #refers to our dictionary to make recommendations
    else:
        #a pair hand
        if player_hand.cards[0].rank == player_hand.cards[1].rank:
            temp_result = sph_dict[((player_hand.cards[0].rank,player_hand.cards[1].rank), dealer_hand.cards[0].rank)]
            print("You should", temp_result + '.')
        #a soft hand
        elif player_hand.cards[0].rank == 'Ace' or player_hand.cards[1].rank == 'Ace':
            if player_hand.cards[0].rank == 'Ace':
                temp_result = sph_dict[((player_hand.cards[0].rank,player_hand.cards[1].rank), dealer_hand.cards[0].rank)]
                print("You should", temp_result + '.')
            else:
                temp_result = sph_dict[((player_hand.cards[1].rank,player_hand.cards[0].rank), dealer_hand.cards[0].rank)]
                print("You should", temp_result + '.')
        #a hard hand
        else:
            temp_result = hard_hands[(str(sum), dealer_hand.cards[0].rank)]
            print("You should", temp_result + '.')
       
def blackjack_setup(played_cards, player_hand):
    
   #card 1 information
    card1_rank = input("Enter your first card's rank.\n")
    card1_suit = input("Enter your first card's suit. \n")
    card1 = card_updater(deck, card1_rank, card1_suit)
    if card1 not in deck.cards:
        print("Your card cannot be found in the deck. Please try again.")
        blackjack_setup(played_cards, player_hand)
    else:
        played_cards.cards.append(card1)
        player_hand.cards.append(card1)
    
    #card 2 information
    card2_rank = input("Enter your second card's rank.\n")
    card2_suit = input("Enter your second card's suit. \n")
    card2 = card_updater(deck, card2_rank, card2_suit)
    if card1.rank == 'Ace' and card2.rank == 'Ace':
        print("You have two aces. Setting the value of an Ace to 1.")
        card1.value = 1
    while card2 not in deck.cards:
        print("Your card cannot be found in the deck. Please try again.\n")
        card2_rank = input("Enter your second card's rank.\n")
        card2_suit = input("Enter your second card's suit. \n")
        card2 = card_updater(deck, card2_rank, card2_suit)
    else:
        played_cards.cards.append(card2)
        player_hand.cards.append(card2)

    #dealer information
    dcard_rank = input("Enter the dealer card rank.\n")
    dcard_suit = input("Enter the dealer card suit. \n")
    dealer = card_updater(deck, dcard_rank, dcard_suit)
    while dealer not in deck.cards:
        print("Your card cannot be found in the deck. Please try again.\n")
        dcard_rank = input("Enter the dealer card rank.\n")
        dcard_suit = input("Enter the dealer card suit. \n")
        dealer = card_updater(deck, dcard_rank, dcard_suit)
    else:
        played_cards.cards.append(dealer)
        dealer_hand.cards.append(dealer)
    
    count2 = counter(played_cards)
    prediction (count2, player_hand, dealer_hand)

    #splitting - will just count each hand as a different round with same dealer card
    if player_hand.cards[0].rank == player_hand.cards[1].rank:
        split_result = input("You have a pair. Would you like to split? Enter Y or N:\n")
        if split_result == 'Y':
            print("This will be your first split hand. The next round will count as the second split hand.")
            player_hand.cards = []
            blackjack_setup(played_cards, player_hand)
        elif split_result == 'N':
            print("Okay.")
        else:
            split_result = input("You have a pair. Would you like to split? Enter Y or N:\n")

    #next action (new card, next round, or end game)
    answer = input('\nAre you going to draw again? Enter Y or N.\n')   

    while answer == 'Y':
        newcard_rank = input("Enter the new card rank.\n")
        newcard_suit = input("Enter the new card suit. \n")
        newcard = card_updater(deck, newcard_rank, newcard_suit)
        if newcard.rank == 'Ace':
            ace_change = input("You have an Ace. Would you like to change its value from 11 to 1? Enter Y or N.\n")
            if ace_change == 'Y':
                newcard.value = 1
                print("Okay.")
            elif ace_change == 'N':
                print("Okay.")
            else:
                ace_change = input("You have an Ace. Would you like to change its value from 11 to 1? Enter Y or N.")
        while newcard not in deck.cards:
            print("Your card cannot be found in the deck. Please try again.\n")
            newcard_rank = input("Enter the new card rank.\n")
            newcard_suit = input("Enter the new card suit. \n")
            newcard = card_updater(deck, newcard_rank, newcard_suit)
        else:
            played_cards.cards.append(newcard)
            player_hand.cards.append(newcard)
        counter3 = counter(played_cards)
        prediction(counter3, player_hand, dealer_hand)
        answer = input('\nAre you going to draw again? Enter Y or N.\n')
    
    if answer == 'N':
        print("\nYou have chosen to end the round.")
        answer2 = input("Are you going to play another round? Enter Y or N.\n")
        if answer2 == 'Y':
            player_hand.cards = []
            blackjack_setup(played_cards, player_hand)
        else:
            print("\n\nOk, thanks for playing.")


#making a deck
deck = Deck()
deck.build()
deck.shuffle()

#creating Deck objects
played_cards = Deck()
player_hand = Deck()
dealer_hand = Deck()

#calling our game
blackjack_setup(played_cards, player_hand)
