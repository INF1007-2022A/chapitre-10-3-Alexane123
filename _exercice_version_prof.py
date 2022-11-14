#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import wave
import struct
import math

import numpy as np
import matplotlib.pyplot as plt


SAMPLING_FREQ = 44100 # Hertz, taux d'échantillonnage standard des CD
SAMPLE_WIDTH = 16 # Échantillons de 16 bit
MAX_INT_SAMPLE_VALUE = 2**(SAMPLE_WIDTH-1) - 1


def merge_channels(channels):
	# Équivalent de :  [sample for samples in zip(*channels) for sample in samples]
	return np.fromiter((sample for samples in zip(*channels) for sample in samples), float)

def separate_channels(samples, num_channels):
	# Équivalent de :  [samples[i::num_channels] for i in range(num_channels)]
	return np.fromiter((samples[i::num_channels] for i in range(num_channels)), float)

def generate_sample_time_points(duration):
	# Générer un tableau de points temporels également espacés en seconde sur la durée donnée
	# On a SAMPLING_FREQ points par seconde, donc duration * SAMPLING_FREQ échantillons
	return np.linspace(0, duration, duration * SAMPLING_FREQ)

def sine(freq, amplitude, duration):
	# Générer une onde sinusoïdale à partir de la fréquence et de l'amplitude donnée, sur le temps demandé et considérant le taux d'échantillonnage.
	# Formule de la valeur y d'une onde sinusoïdale à l'angle x en fonction de sa fréquence F et de son amplitude A :
	# y = A * sin(F * x), où x est en radian.
	# Si on veut le x qui correspond au moment t, on peut dire que 2π représente une seconde, donc x = t * 2π,
	# Or t est en secondes, donc t = i / nb_échantillons_par_secondes, où i est le numéro d'échantillon.

	# y = A * sin(F * 2π*t)
	time_points = generate_sample_time_points(duration)
	return amplitude * np.sin(freq * 2*np.pi * time_points)

def square(freq, amplitude, duration):
	# Générer une onde carrée d'une fréquence et amplitude donnée.
	# y = A * sgn(sin(F * 2π*t))
	return amplitude * np.sign(sine(freq, 1, duration))

def sawtooth(freq, amplitude, duration):
	# Générer une onde en dents de scie (sawtooth) à partir de l'amplitude et fréquence donnée.
	# La formule d'une onde en dents de scie à un temps t en fonction de la fréquence F et de l'amplitude A :
	# y = A * 2(t * F - floor(1/2 + t * F))
	t = generate_sample_time_points(duration)
	return amplitude * 2 * (t * freq - np.floor(1.0/2.0 + t * freq))

def sine_with_overtones(root_freq, amplitude, overtones, duration):
	# Générer une onde sinusoïdale avec ses harmoniques. Le paramètre overtones est une liste de tuple où le premier élément est le multiple de la fondamentale et le deuxième élément est l'amplitude relative de l'harmonique.
	# On bâtit un signal avec la fondamentale
	signal = sine(root_freq, amplitude, duration)
	# Pour chaque harmonique (overtone en anglais), on a un facteur de fréquence et un facteur d'amplitude :
	for freq_factor, amp_factor in overtones:
		# Construire le signal de l'harmonique en appliquant les deux facteurs.
		overtone = sine(root_freq * freq_factor, amplitude * amp_factor, duration)
		# Ajouter l'harmonique au signal complet.
		np.add(signal, overtone, out=signal)
	return signal

def normalize(samples, norm_target):
	# Normalisez un signal à l'amplitude donnée
	# 1. il faut trouver l'échantillon le plus haut en valeur absolue
	abs_samples = np.abs(samples)
	max_sample = np.max(abs_samples)
	# 2. Calcule coefficient entre échantillon max et la cible
	coeff = norm_target / max_sample
	# 3. Applique mon coefficient
	normalized_samples = coeff * samples
	return normalized_samples

def convert_to_bytes(samples):
	# Convertir les échantillons en tableau de bytes en les convertissant en entiers 16 bits.
	# Les échantillons en entrée sont entre -1 et 1, nous voulons les mettre entre -MAX_INT_SAMPLE_VALUE et MAX_INT_SAMPLE_VALUE
	# Juste pour être certain de ne pas avoir de problème, on doit clamper les valeurs d'entrée entre -1 et 1.

	# Limiter (ou clamp/clip) les échantillons entre -1 et 1
	clipped = np.clip(samples, -1, 1)
	# Convertir en entier 16-bit signés. Le < veut dire little endian, i2 veut dire entier signé à deux octets (16-bit).
	int_samples = (clipped * MAX_INT_SAMPLE_VALUE).astype("<i2")
	# Convertir en bytes
	sample_bytes = int_samples.tobytes()
	return sample_bytes

def convert_to_samples(bytes):
	# Faire l'opération inverse de convert_to_bytes, en convertissant des échantillons entier 16 bits en échantillons réels
	# 1. Convertir en numpy array du bon type (entier 16 bit signés)
	int_samples = np.frombuffer(bytes, dtype="<i2")
	# 2. Convertir en réel dans [-1, 1]
	samples = int_samples.astype(float) / MAX_INT_SAMPLE_VALUE


def main():
	try:
		# On met les fichiers de sortie dans leur propre dossier.
		os.mkdir("output")
	except:
		pass

	# On affiche une onde d'exemple 
	xs = generate_sample_time_points(3)
	ys = square(1.0, 0.5, 3) + sawtooth(10.0, 0.1, 3)
	plt.figure(figsize=(12, 6))
	plt.plot(xs, ys)
	plt.grid(color="wheat")
	plt.ylim([-1.1, 1.1])
	plt.xlim([0, 2])
	plt.xlabel("t (s)")
	plt.ylabel("y")
	plt.show()

	# Exemple d'un la et mi (quinte juste), un dans le channel gauche et l'autre dans le channel droit
	with wave.open("output/perfect_fifth_panned.wav", "wb") as writer:
		# On fait la config du writer (2 channels, échantillons de deux octets, fréquence d'échantillonnage).
		writer.setnchannels(2)
		writer.setsampwidth(2)
		writer.setframerate(SAMPLING_FREQ)

		# On génére un la3 (220 Hz) et un mi4 (intonation juste, donc ratio de 3/2)
		#samples1 = sine(220, 0.9, 30.0)
		samples1 = sawtooth(220, 0.9, 30.0)
		#samples2 = sine(220 * (3/2), 0.7, 30.0)
		samples2 = sawtooth(220 * (3/2), 0.7, 30.0)

		# On met les samples dans des channels séparés (la à gauche, mi à droite)
		merged = merge_channels([samples1, samples2])
		data = convert_to_bytes(merged)

		writer.writeframes(data)

	with wave.open("output/major_chord.wav", "wb") as writer:
		writer.setnchannels(1)
		writer.setsampwidth(2)
		writer.setframerate(SAMPLING_FREQ)

		# Un accord majeur (racine, tierce, quinte, octave) en intonation juste
		root_freq = 220
		root = sine(root_freq, 1, 10.0)
		third = sine(root_freq * 5/4, 1, 10.0)
		fifth = sine(root_freq * 3/2, 1, 10.0)
		octave = sine(root_freq * 2, 1, 10.0)
		# Étant donné qu'on additionne les signaux, on normalize pour que ça soit à un bon niveau.
		chord = normalize(root + third + fifth + octave, 0.89)

		writer.writeframes(convert_to_bytes(chord))

	with wave.open("output/overtones.wav", "wb") as writer:
		writer.setnchannels(1)
		writer.setsampwidth(2)
		writer.setframerate(SAMPLING_FREQ)

		# On génère un signal avec ses 3 premières harmoniques (2x, 3x et 4x la fréquence de base)
		samples = sine_with_overtones(220, 1, [(i, 0.15**(i-1)) for i in range(2, 5)], 10)
		# Encore, on additionne des signaux, donc on normalise.
		samples = normalize(samples, 0.89)

		writer.writeframes(convert_to_bytes(samples))

if __name__ == "__main__":
	main()
