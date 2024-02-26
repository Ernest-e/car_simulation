#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:22:11 2024

@author: amina
"""

import pygame
import math
import numpy as np
import random
import torch
import torch.nn as nn
from deap import base, algorithms
from deap import creator
from deap import tools
from deap.algorithms import varAnd
import matplotlib.pyplot as plt
import time


RANDOM_SEED = 42
random.seed(RANDOM_SEED)



class Obstacle:
    def __init__(self, position, size):
        self.position = position  # Положение препятствия (x, y)
        self.size = size  # Размер препятствия



# Класс для трассы
class Track:
    def __init__(self, points, widths):
        self.points = points
        self.widths = widths


# Класс для автомобиля
class Car:
    def __init__(self, x, y, angle=0, speed = 0, acceleration = 0, width=10, length=10, max_speed=2):
        self.init_x = x
        self.init_y = y
        self.init_angle = angle
        self.init_speed = speed
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.acceleration = acceleration
        self.width = width
        self.length = length
        self.position = (self.x, self.y)
        self.max_speed = max_speed

    def update(self):
        self.speed = max(0, min(self.speed, self.max_speed))
        
        self.x += math.cos(math.radians(self.angle%360)) * self.speed
        self.y += math.sin(math.radians(self.angle%360)) * self.speed
        

    def reset(self, x, y):
        """
        Сброс состояния автомобиля к исходным параметрам.
        """
        self.init_x = x
        self.init_y = y
        self.x = self.init_x
        self.y = self.init_y
        self.angle = self.init_angle
        self.speed = self.init_speed
        
    

    def draw(self, screen):
        pygame.draw.polygon(screen, GREEN, [(self.x + math.cos(math.radians(self.angle)) * self.length, self.y + math.sin(math.radians(self.angle)) * self.length),
                                            (self.x + math.cos(math.radians(self.angle + 120)) * self.width, self.y + math.sin(math.radians(self.angle + 120)) * self.width),
                                            (self.x + math.cos(math.radians(self.angle + 240)) * self.width, self.y + math.sin(math.radians(self.angle + 240)) * self.width)])




def find_intersection(ray_origin, ray_direction, point1, point2):
    # Вектор направления луча
    v1 = ray_origin - point1
    v2 = point2 - point1
    v3 = np.array([-ray_direction[1], ray_direction[0]])

    dot_product = np.dot(v2, v3)
    if dot_product == 0:
        return None  # Векторы параллельны, нет пересечения

    t1 = np.cross(v2, v1) / dot_product
    t2 = np.dot(v1, v3) / dot_product

    if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
        return ray_origin + t1 * ray_direction
    return None

def lidar_scan(car_pos, car_angle, track_points, track_widths, num_rays=4, max_distance=100):
    """
    Симулирует лидар, сканируя окружающую среду на наличие границ трассы.
    """
    angles = np.linspace(0, 360, num_rays, endpoint=False) + car_angle
    distances = []

    for angle in angles:
        # Направление луча с учетом угла автомобиля
        ray_dir = np.array([math.cos(math.radians(angle%360)), math.sin(math.radians(angle%360))])
        ray_origin = np.array(car_pos)
        closest_intersection = None

        for i in range(len(track_points) - 1):
            point1 = np.array(track_points[i])
            point2 = np.array(track_points[i + 1])

            # Находим пересечения для обеих сторон трассы
            for offset in [-track_widths[i] / 2, track_widths[i] / 2]:
                offset_vec = np.array([point2[1] - point1[1], point1[0] - point2[0]])
                offset_vec = offset * offset_vec / np.linalg.norm(offset_vec)

                intersection = find_intersection(ray_origin, ray_dir, point1 + offset_vec, point2 + offset_vec)
                if intersection is not None:
                    if closest_intersection is None or np.linalg.norm(intersection - ray_origin) < np.linalg.norm(closest_intersection - ray_origin):
                        closest_intersection = intersection

        if closest_intersection is not None:
            distances.append(np.linalg.norm(closest_intersection - ray_origin))
        else:
            distances.append(max_distance)

    return [distance / max_distance for distance in distances] # нормализация


# Получения информации о состоянии машины относительно трассы
def get_car_state(car, track):
    max_speed = 2
    car_pos = car.x, car.y
    car_angle = car.angle / 360
    car_speed = car.speed / max_speed
    track_points = track.points
    track_widths = track.widths
    distances = lidar_scan(car_pos, car_angle, track_points, track_widths)
    distances.extend([car_angle, car_speed])

    return torch.tensor(distances, dtype=torch.float32)
    



# Нейронная сеть
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(6, 10)  # Пример: 6 входов, 10 нейронов в скрытом слое
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(10, 2)  # 2 выхода: ускорение и поворот

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class NNetwork:
    """Многослойная полносвязная нейронная сеть прямого распространения"""

    @staticmethod
    def getTotalWeights(*layers):
        return sum([(layers[i]+1)*layers[i+1] for i in range(len(layers)-1)])

    def __init__(self, inputs, *layers):
        self.layers = []        # список числа нейронов по слоям
        self.acts = []          # список функций активаций (по слоям)

        # формирование списка матриц весов для нейронов каждого слоя и списка функций активации
        self.n_layers = len(layers)
        for i in range(self.n_layers):
            self.acts.append(self.act_relu)
            if i == 0:
                self.layers.append(self.getInitialWeights(layers[0], inputs+1))         # +1 - это вход для bias
            else:
                self.layers.append(self.getInitialWeights(layers[i], layers[i-1]+1))    # +1 - это вход для bias

        self.acts[-1] = self.act_tanh     #последний слой имеет пороговую функцию активакции

    def getInitialWeights(self, n, m):
        return np.random.triangular(-1, 0, 1, size=(n, m))

    def act_relu(self, x):
        x[x < 0] = 0
        return x
    
    def act_leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)

    def act_th(self, x):
        x[x > 0] = 1
        x[x <= 0] = 0
        return x
    
    def act_tanh(self, x):
        return np.tanh(x)
    
    def act_linear(self, x):
        return x

    def get_weights(self):
        return np.hstack([w.ravel() for w in self.layers])

    def set_weights(self, weights):
        off = 0
        for i, w in enumerate(self.layers):
            w_set = weights[off:off+w.size]
            off += w.size
            self.layers[i] = np.array(w_set).reshape(w.shape)

    def predict(self, inputs):
        f = inputs
        for i, w in enumerate(self.layers):
            f = np.append(f, 1.0)       # добавляем значение входа для bias
            f = self.acts[i](w @ f)

        return f


def neural_network_decision(state, model):
    """
    Принимает решение на основе текущего состояния автомобиля и модели нейронной сети.
    
    :param car: объект автомобиля
    :param model: экземпляр обученной модели нейронной сети
    :return: speed_change (изменение скорости), angle_change (изменение угла)
    """
    
    # Отключаем расчет градиентов, так как мы находимся в режиме инференса
    # with torch.no_grad():
        # Пропускаем состояние автомобиля через нейронную сеть
    output = model.predict(state)
    
    # Извлекаем желаемые изменения скорости и угла из выходных данных сети
    speed_change, angle_change = output[0], output[1]
    
    return speed_change, angle_change



def point_to_line_distance(point, line_start, line_end):
    """Вычисляет расстояние от точки до отрезка линии."""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    den = math.sqrt((y2-y1)**2 + (x2-x1)**2)
    return num / den if den else 0

def is_point_on_segment(point, start, end, buffer=0):
    """
    Проверяет, находится ли точка на отрезке, учитывая небольшой буфер (ширину трассы).
    """
    # Расширяем проверку на наличие в пределах прямоугольника, образованного начальной и конечной точками сегмента.
    px, py = point
    sx, sy = start
    ex, ey = end

    # Учитываем буфер в расчетах
    dx = ex - sx
    dy = ey - sy

    if dx == 0 and dy == 0:  # Сегмент представляет собой точку
        return math.sqrt((px - sx) ** 2 + (py - sy) ** 2) <= buffer

    # Нормализуем сегмент
    norm = dx * dx + dy * dy
    u = ((px - sx) * dx + (py - sy) * dy) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # Находим ближайшую точку на сегменте к точке
    x = sx + u * dx
    y = sy + u * dy

    dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
    return dist <= buffer


def is_car_off_track(car, track):
    for i in range(len(track.points) - 1):
        if is_point_on_segment((car.x, car.y), track.points[i], track.points[i + 1], buffer=track.widths[i] / 2):
            return False  # Автомобиль находится на трассе
    return True  # Автомобиль вне трассы




def line_intersection(line1, line2):
    """
    Определяет точку пересечения двух линий (если она существует).
    
    :param line1: Кортеж из двух точек (x1, y1), (x2, y2), определяющих первую линию.
    :param line2: Кортеж из двух точек (x3, y3), (x4, y4), определяющих вторую линию.
    :return: Координаты точки пересечения (x, y) или None, если линии не пересекаются.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None  # Линии параллельны

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

    if 0 <= t <= 1 and 0 <= u <= 1:
        # Точка пересечения находится в пределах обоих отрезков
        return x1 + t * (x2 - x1), y1 + t * (y2 - y1)

    return None

def has_finished(car_position, prev_car_position, finish_line):
    """
    Проверяет, пересекла ли машина линию финиша между двумя последовательными положениями.
    
    :param car_position: Текущее положение машины (x, y).
    :param prev_car_position: Предыдущее положение машины (x, y).
    :param finish_line: Координаты линии финиша (x1, y1, x2, y2).
    :return: True, если машина пересекла линию финиша; иначе False.
    """
    intersection = line_intersection((prev_car_position[0], prev_car_position[1], car_position[0], car_position[1]), finish_line)
    return intersection is not None






def simulate_drive(network_weights, car, track, max_time=60):
    time_step = 0.1  # Длительность одного шага времени в секундах.
    total_time = 0  # Общее время симуляции.
    total_distance = 0  # Инициализация общего пройденного расстояния
    
    finish_line = (track.points[-1][0], track.points[-1][1] - track.widths[-1]/2, track.points[-1][0], track.points[-1][1] + track.widths[-1]/2)

    simulation_ended_early = False
    while total_time < max_time:
        prev_car_position = car.x, car.y
        car_state = get_car_state(car, track)
        speed_change, angle_change = neural_network_decision(car_state, network_weights)
        car.speed += speed_change
        car.angle += angle_change
        car.update()
        new_car_position = car.x, car.y

        
        if has_finished(new_car_position, prev_car_position, finish_line):
            print('FINISH!!!!!!!')
            break
        
        if is_car_off_track(car, track):
            print('CRASHHHHHH CAR IS OFF TRACK!!!!!!!!!!!')
            simulation_ended_early = True
            break  # Прерываем симуляцию, если автомобиль вышел за пределы трассы.
            
        print(car.x, car.y)
        total_distance += car.speed * time_step  # Увеличиваем пройденное расстояние
        total_time += time_step
    
    return total_distance, total_time, simulation_ended_early







def create_checkpoints(track_points, track_widths):
    checkpoints = []
    for i in range(len(track_points)-1):
        p1 = np.array(track_points[i])
        p2 = np.array(track_points[i+1])
        
        # Находим середину отрезка между двумя точками
        mid_point = (p1 + p2) / 2
        direction = p2 - p1
        normal = np.array([-direction[1], direction[0]])  # Нормаль к направлению
        normal = normal / np.linalg.norm(normal)  # Нормализация
        
        # Создаем две точки чекпоинта, перпендикулярно направлению трассы
        checkpoint_start = mid_point + normal * track_widths[i] / 2
        checkpoint_end = mid_point - normal * track_widths[i] / 2
        checkpoints.append((checkpoint_start, checkpoint_end))
        
    return checkpoints

def check_checkpoint_crossing(car_position, prev_car_position, checkpoints):
    car_pos = np.array(car_position)
    prev_car_pos = np.array(prev_car_position)
    
    for checkpoint in checkpoints:
        cp_start, cp_end = checkpoint
        # Проверяем пересечение линии автомобиля и линии чекпоинта
        if line_intersection((cp_start[0],cp_start[1], cp_end[0], cp_end[1]), (car_pos[0],car_pos[1], prev_car_pos[0], prev_car_pos[1])):
            return True
    return False

track_points = [
    (90, 150),  # Старт
    (300, 150),  # Прямой участок
    (500, 300),  # Плавный поворот вправо
    (500, 500),  # Прямой участок вниз
    (300, 650),  # Плавный поворот влево
    (100, 500),  # Возвращение к старту
    (100, 150)   # Замыкание трассы
]
track_widths = [50] * len(track_points)  # Ширина трассы



# track = Track(track_points, track_widths)
# finish_line = [(track.points[-1][0], track.points[-1][1] - track.widths[-1]/2, track.points[-1][0], track.points[-1][1] + track.widths[-1]/2)]
# car = Car(start_point[0], start_point[1])



NEURONS_IN_LAYERS = [6,10, 2]               # распределение числа нейронов по слоям (первое значение - число входов)
network = NNetwork(*NEURONS_IN_LAYERS)

LENGTH_CHROM = NNetwork.getTotalWeights(*NEURONS_IN_LAYERS)    # длина хромосомы, подлежащей оптимизации
LOW = -1.0
UP = 1.0
ETA = 20

# константы генетического алгоритма
POPULATION_SIZE = 20   # количество индивидуумов в популяции
P_CROSSOVER = 0.9       # вероятность скрещивания
P_MUTATION = 0.1      # вероятность мутации индивидуума
MAX_GENERATIONS = 30    # максимальное количество поколений
HALL_OF_FAME_SIZE = 2

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)



creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
toolbox.register("randomWeight", random.uniform, -1.0, 1.0)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.randomWeight, LENGTH_CHROM)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)



def eaSimpleElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, callback=None):
    """Перелеланный алгоритм eaSimple с элементом элитизма
    """

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if callback:
            callback[0](*callback[1])

    return population, logbook


def generate_start_positions(track):
    positions = []
    for i in range(len(track.points)-1):
        start_point = track.points[i]
        end_point = track.points[i+1]
        width = track.widths[i]

        # Рассчитываем вектор от текущей точки к следующей
        vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
        vector_length = (vector[0]**2 + vector[1]**2)**0.5

        # Количество стартовых позиций зависит от длины участка
        num_positions = int(vector_length / width)

        for j in range(num_positions):
            # Интерполируем позиции между текущей и следующей точками
            ratio = j / num_positions
            new_position = (start_point[0] + vector[0] * ratio, start_point[1] + vector[1] * ratio)
            positions.append(new_position)

    return positions



def getScore(individual):
    network.set_weights(individual)
    
    track = Track(track_points, track_widths)
    start_positions = generate_start_positions(track)
    start_pos = random.choice(start_positions)
    car = Car(*start_pos)
    
    time_step = 0.1  # Длительность одного шага времени в секундах.
    total_time = 0  # Общее время симуляции.
    total_distance = 0  # Инициализация общего пройденного расстояния
    off_track_penalty = 0
    checkpoints_reward = 0
    # time_since_last_movement = 0
    # stop_time_threshold = 0.2
    stop_penalty=0
    circular_motion_penalty = 0 
    last_checkpoint_distance = 0.0  # Расстояние до последней контрольной точки
    checkpoints_passed = 0 
    
    circular_motion_threshold = 400
    restarts_count = 0  # Счетчик рестартов
    
    checkpoints = create_checkpoints(track.points, track.widths)
    finish_line = (track.points[-1][0], track.points[-1][1] - track.widths[-1]/2, track.points[-1][0], track.points[-1][1] + track.widths[-1]/2)
    while total_time<60:
        prev_car_position = car.x, car.y
        car_state = get_car_state(car, track)
        # print(car_state)
        speed_change, angle_change = neural_network_decision(car_state, network)
        car.speed += speed_change
        car.angle += angle_change
        car.update()
        new_car_position = car.x, car.y
        # print(car.x, car.y, angle_change)
        
        # Проверка на остановку
        if np.linalg.norm(np.array(new_car_position) - np.array(prev_car_position)) < 0.01:  # Почти не двигался
            stop_penalty -= 500   # Накопление штрафов за отсутствие движения
            
        
        # Проверка на круговое движение без прохождения контрольных точек
        distance_travelled = np.linalg.norm(np.array(new_car_position) - np.array(start_pos)) - last_checkpoint_distance
        if distance_travelled > circular_motion_threshold and checkpoints_passed == 0:
            circular_motion_penalty -= 200  # Штраф за круговое движение без достижения прогресса
            # Обнуляем дистанцию, чтобы предотвратить повторное наложение штрафа без изменения поведения
            last_checkpoint_distance = np.linalg.norm(np.array(new_car_position) - np.array(start_pos))
            
        
        if check_checkpoint_crossing(new_car_position, prev_car_position, checkpoints):
            checkpoints_reward += 300
            checkpoints_passed += 1  # Увеличиваем счётчик контрольных точек
            last_checkpoint_distance = np.linalg.norm(np.array(new_car_position) - np.array(start_pos))  # Обновляем расстояние до последней контрольной точки

    
        
        # if has_finished(new_car_position, prev_car_position, finish_line):
        #     checkpoints_reward += 100  # Большая награда за достижение финиша
        #     print('FINISH!!!')
        #     break
        
        if is_car_off_track(car, track):
            off_track_penalty -= 100  # Штраф за выход за пределы трассы
            restarts_count += 1
            print('Reset')
            start_pos = random.choice(start_positions)
            car.reset(*start_pos)  # Перезапуск с новой случайной стартовой позиции
            continue
            
        # print(car.x, car.y)
        total_distance += np.linalg.norm(np.array(new_car_position) - np.array(prev_car_position))  # Увеличиваем пройденное расстояние
        total_time += time_step
        
    print('дистанция', total_distance)
    
    restarts_penalty = -250 * restarts_count 
    fitness = total_distance  + checkpoints_reward + stop_penalty + off_track_penalty + circular_motion_penalty + restarts_penalty #- (total_time * 0.5)
    print(fitness)
    return fitness,

toolbox.register("evaluate", getScore)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0/LENGTH_CHROM)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

population, logbook = eaSimpleElitism(population, toolbox,
                                        cxpb=P_CROSSOVER,
                                        mutpb=P_MUTATION,
                                        ngen=MAX_GENERATIONS,
                                        halloffame=hof,
                                        stats=stats,
                                        verbose=True)

#%%
track = Track(track_points, track_widths)
generate_start_positions(track)

#%%
maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")


best = hof.items[0]
print(best)

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()
#%%
print(network.get_weights())
print(best)
try_model = NNetwork(*NEURONS_IN_LAYERS)
try_model.set_weights(best)
try_model.get_weights()
#%%

# Инициализация Pygame
pygame.init()

track_points = [
    (100, 150),  # Старт
    (300, 150),  # Прямой участок
    (500, 300),  # Плавный поворот вправо
    (500, 500),  # Прямой участок вниз
    (300, 650),  # Плавный поворот влево
    (100, 500),  # Возвращение к старту
    (100, 150)   # Замыкание трассы
]

# Параметры экрана
screen_width, screen_height = 1000, 1000
screen = pygame.display.set_mode((screen_width, screen_height))
font = pygame.font.Font(None, 36) # None используется для стандартного шрифта, 36 - размер шрифта

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def draw_track(screen, track):
    for i in range(len(track.points) - 1):
        start_pos = track.points[i]
        end_pos = track.points[i + 1]
        pygame.draw.line(screen, WHITE, start_pos, end_pos, track.widths[i])

start_point = track_points[0]
end_point = track_points[-1]

# Главный цикл игры
track = Track(track_points, track_widths)
car = Car(start_point[0], start_point[1])
finish_line = (track.points[-1][0], track.points[-1][1] - track.widths[-1]/2, track.points[-1][0], track.points[-1][1] + track.widths[-1]/2)
print(finish_line)
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    prev_car_position = car.x, car.y
    print(prev_car_position)
    car_state = get_car_state(car, track)
    speed_change, angle_change = neural_network_decision(car_state, network)
    print('angel', angle_change)
    car.speed += speed_change
    print('speed', car.speed)
    car.angle += angle_change
    car.update()
    new_car_position = car.x, car.y
    print(new_car_position)
    # if has_finished(new_car_position, prev_car_position, finish_line):
    #       text = font.render("FINISH!", True, WHITE, BLACK) 
    #       text_rect = text.get_rect()
    #       text_rect.center = (500, 500) # Размещаем текст в центре экрана
    #       screen.blit(text, text_rect)
    #       pygame.display.flip()
    #       print('finish')
    #       time.sleep(5)
    #       break
        
    # if is_car_off_track(car, track):
    #       text = font.render("CRASH!", True, WHITE, BLACK) 
    #       text_rect = text.get_rect()
    #       text_rect.center = (500, 500) # Размещаем текст в центре экрана
    #       screen.blit(text, text_rect)
    #       pygame.display.flip()
    #       print('crash')
    #       time.sleep(5)
    #       break  # Или продолжите симуляцию с последней валидной позиции

    screen.fill(BLACK)
    draw_track(screen, track)
    # draw_obstacles(screen)
    car.draw(screen)
    pygame.display.flip()

    pygame.time.delay(30)

pygame.quit()










# Трасса и препятствия
# track_points = [(100, 150), (200, 150), (300, 100), (500, 300), (500, 400), (600, 500), (700, 400), (800, 300), (900, 200)]
# track_widths = [50, 50, 50, 50, 50, 50, 50, 50, 50]

# start_point = track_points[0]
# end_point = track_points[-1]


# track = Track(track_points, track_widths)
# finish_line = [(track.points[-1][0], track.points[-1][1] - track.widths[-1]/2, track.points[-1][0], track.points[-1][1] + track.widths[-1]/2)]
# car = Car(start_point[0], start_point[1])



# population_size = 50
# generations = 100

# initial_population = create_initial_population(population_size, NeuralNetwork)
# best_model = genetic_algorithm(initial_population, generations, car, track)
# print(best_model)


# simulate_drive(best_model, car, track)



# model = NeuralNetwork()

# simulate_drive(model, car, track)
# state1 = get_car_state(car, track)
# move = model(state1)
# print(move)
# car.speed = move[0].item()
# car.angle = move[1].item()
# car.update()
# print(car.x, car.y)
# print(is_car_off_track(car, track))

# obstacles = [(300, 90, 25), (490, 200, 30)]  # Координаты x, y и размер






# def draw_track(screen, track):
#     for i in range(len(track_points) - 1):
#         start_pos = track.points[i]
#         end_pos = track.points[i + 1]
#         pygame.draw.line(screen, WHITE, start_pos, end_pos, track.widths[i])

        

# # Функция для отрисовки препятствий
# def draw_obstacles(screen):
#     for obstacle in obstacles:
#         pygame.draw.circle(screen, RED, (obstacle[0], obstacle[1]), obstacle[2])







# def fitness_function(distance, time, off_track_penalty):
#     """
#     Функция оценки для определения "фитнеса" автомобиля.

#     :param distance: Пройденное расстояние по трассе (в пикселях или метрах).
#     :param time: Затраченное время на прохождение трассы (в секундах).
#     :param off_track_penalty: Штраф за выход за пределы трассы (количество инцидентов).
    
#     :return: Значение фитнеса для данного прохождения.
#     """
#     # Параметры для настройки важности каждого фактора
#     distance_weight = 1.0
#     time_weight = -0.5  # Минус, потому что меньшее время - лучше
#     penalty_weight = -20.0  # Штраф за выход за пределы трассы

#     # Вычисление фитнеса
#     fitness = (distance * distance_weight) + (time * time_weight) + (off_track_penalty * penalty_weight)

#     return fitness


# def fitness(nn, inputs, expected_output):
#     output = nn.forward(inputs)
#     # Простая мера приспособленности: насколько близко выход нейросети к ожидаемому значению
#     fitness = -np.sum((output - expected_output) ** 2)
#     return fitness



















# # Пример использования
# network_shape = [(5, 10), (10, 2)]  # Пример структуры нейронной сети
# population_size = 50
# generations = 100

# initial_population = create_initial_population(population_size, network_shape)
# best_weights = genetic_algorithm(initial_population, generations, network_shape)

# # Загрузите лучшие веса в вашу модель нейронной сети







# # Главный цикл игры
# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     keys = pygame.key.get_pressed()
#     if keys[pygame.K_LEFT]:
#         car.angle += 5
#     if keys[pygame.K_RIGHT]:
#         car.angle -= 5
#     if keys[pygame.K_UP]:
#         car.speed += 0.1
#     if keys[pygame.K_DOWN]:
#         car.speed -= 0.1

#     car.update()

#     screen.fill(BLACK)
#     draw_track(screen, track)
#     draw_obstacles(screen)
#     car.draw(screen)
#     pygame.display.flip()

#     pygame.time.delay(30)

# pygame.quit()

