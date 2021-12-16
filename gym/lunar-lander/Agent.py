"""Schrijf nu een functie die train heet.
Dit is een functie van de agent die het policy-network update adhv het Double Deep Q-Learning algoritme.
Dit is een variatie op de Bellman Optimality Equation.
Het idee is dat je hierin targets aanmaakt voor je netwerk (je labels) dit gaat in de volgende stappen:

-Pak een batch transities uit het geheugen. De grootte van de batch is een hyperparameter
-Pak de next_state uit je transities en bereken adhv je policy netwerk wat de beste actie is
-Pak deze beste actie en bereken met je target netwerk wat de q-waarde van deze actie in next_state is.
Onthoud dat als next_state een terminal state is, alle q-waardes 0 moeten zijn.
-Bereken nu de target: reward + gamma * resultaat vorige stap
-Doe stap 2-5 voor alle transities in de batch
-Voor gradient descent uit alleen op de outputnodes van je netwerk die representatief zijn voor acties die daadwerkelijk genomen zijn.
-In pytorch kan je selecteren op welke output nodes je gradient descent wilt toepassen
-Als je library dit niet toestaat kan je zorgen dat je target een lijst is waarin alleen de index van de genomen actie
verandert naar de target berekend in stap 4
---------
"""


class Agent:
    def __init__(self, gamma: float, alpha: float, tau: float, epsilon: float, batchsize: int, learning_rate: float):
        """ Er zijn nog allerlei eigenschappen van de agent niet gespecificeerd. Onder andere Gamma/Alpha/Batchsize//epsilon etc.
        Bedenk zelf waar die terecht moeten aan de hand van je theoretische begrip van het onderwerp."""
        pass

    def train(self):
        """Train a network with the Double Deep Q-Learning algorithm"""
        pass

    def copy_model(self):
        """Schrijf een functie die copy_model heet, deze voegt de policy en het target-network samen.
            Tau is hiervoor een input parameter die bepaalt hoeveel procent van het target-netwerk vervangen wordt door het policy netwerk.
            We doen dit minder vaak dan het trainen van het policy netwerk.
            Maar dit gebeurt wel om de n stappen. N is een optimaliseerbare hyperparameter"""
        pass
