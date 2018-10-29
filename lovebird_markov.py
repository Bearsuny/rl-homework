import sys
import os

import pygame
from pygame.locals import *

import numpy as np
from collections import Iterable


class GameConfig:
    caption = 'lovebird-ff'
    window_size = (960, 540)
    grid_size = (60, 60)
    brick_size = (138, 793)
    bird_size = (780, 690)
    fps = 30
    save_number = 0

    loop_flag = False

    root_path = '.'
    resource_path = 'assets'
    save_path = 'output'


class GameItem(pygame.sprite.Sprite):
    def __init__(self, source_path, size):
        super().__init__()
        self.image = pygame.image.load(source_path).convert_alpha()
        self.image = pygame.transform.scale(self.image, size)
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)

    def update_rect(self):
        self.rect = self.image.get_rect()

    def update_mask(self):
        self.mask = pygame.mask.from_surface(self.image)

    def blit(self, screen):
        screen.blit(self.image, self.rect)


class Grid():
    def __init__(self, color, width, mode):
        self.containers = []
        self.color = color
        self.width = width
        self.mode = mode
        if mode == 'rect':
            self.text_font = pygame.font.SysFont('times', 15)
            self.text_surface = self.text_font.render(str(0), True, (0, 0, 0))

    def blit(self, screen):
        if self.mode == 'line':
            for item in self.containers:
                pygame.draw.line(screen, self.color, *item, self.width)
        if self.mode == 'rect':
            for i, item in enumerate(self.containers):
                pygame.draw.rect(screen, self.color, item, self.width)
                self.text_surface = self.text_font.render(str(i), True, (0, 0, 0))
                pos = item[0]
                screen.blit(self.text_surface, (pos[0] + 5, pos[1] + 5))


class Brick(GameItem):
    def __init__(self, source_path, source_size, grid_size):
        size = (grid_size[0], source_size[1] * grid_size[1] // source_size[0])
        super().__init__(source_path, size)

    def chop_with_flip(self, height, flip_flag, location):
        self.image = pygame.transform.chop(self.image, [0, self.image.get_size()[1] - height, 0, height]).convert_alpha()
        if flip_flag:
            self.image = pygame.transform.flip(self.image, False, True).convert_alpha()
        self.update_rect()
        self.rect[0], self.rect[1] = location
        self.update_mask()


class Bird(GameItem):
    def __init__(self, source_path, source_size, gender, window_size):
        super().__init__(source_path, source_size)
        self.gender = gender
        if gender == 'female':
            self.image = pygame.transform.flip(self.image, True, False).convert_alpha()
            self.update_rect()
            self.rect[0] = window_size[0]-self.image.get_size()[0]
            self.update_mask()

    def update(self, step, direction):
        if direction == 0:  # Down
            self.rect.move_ip(0, step)
        if direction == 1:  # Right
            self.rect.move_ip(step, 0)
        if direction == 2:  # Up
            self.rect.move_ip(0, -step)
        if direction == 3:  # Left
            self.rect.move_ip(-step, 0)

    def collision(self, step, bricks, window_size):
        directions = [(0, step), (step, 0), (0, -step), (-step, 0)]
        rects = [self.rect.move(item) for item in directions]
        last_rect = self.rect
        candidates = []
        for i, item in enumerate(rects):
            self.rect = item
            collides = pygame.sprite.spritecollide(self, bricks, False, pygame.sprite.collide_mask)
            if collides:
                pass
            else:
                if 0 <= self.rect[0] <= window_size[0]-self.image.get_size()[0] and 0 <= self.rect[1] <= window_size[1]-self.image.get_size()[1]:
                    candidates.append(directions[i])
        choice = np.random.choice(len(candidates))
        direction = directions.index(candidates[choice])
        self.rect = last_rect
        return direction

    def find_mate(self, bird):
        return True if pygame.sprite.collide_mask(self, bird) else False

    def reset(self):
        self.rect[0], self.rect[1] = 0, 0


class LoveBirdGame():
    def __init__(self, caption, window_size):
        pygame.init()
        pygame.display.set_caption(caption)
        self.screen = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()
        self.history_grid = Grid((255, 0, 0), 5, 'rect')

    def loop(self, bg, grid, bricks, birds, mode):
        while True:
            self.clock.tick(GameConfig.fps)

            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_RETURN:
                        GameConfig.loop_flag = not GameConfig.loop_flag
                    if event.key == K_ESCAPE:
                        pygame.image.save(self.screen, os.path.join(GameConfig.root_path, GameConfig.save_path, f'{GameConfig.save_number}.png'))
                        GameConfig.save_number += 1
                elif event.type == QUIT:
                    pygame.quit()
                    sys.exit()

            male_bird, female_bird = [bird for bird in birds]
            pos = ((male_bird.rect[0], male_bird.rect[1]), GameConfig.grid_size)
            if pos not in self.history_grid.containers:
                self.history_grid.containers.append(pos)

            self.blit(bg, grid, bricks, birds, self.history_grid)

            if GameConfig.loop_flag:
                action = male_bird.collision(GameConfig.grid_size[0], bricks, GameConfig.window_size)
                male_bird.update(GameConfig.grid_size[0], action)
                if male_bird.find_mate(female_bird):
                    GameConfig.loop_flag = not GameConfig.loop_flag
                    self.reset()
                    male_bird.reset()

            pygame.display.update()

    def blit(self, *objs):
        for obj in objs:
            if isinstance(obj, Iterable):
                for item in obj:
                    item.blit(self.screen)
            else:
                obj.blit(self.screen)

    def reset(self):
        self.history_grid.containers = []


def game_env_init():
    game = LoveBirdGame(GameConfig.caption, GameConfig.window_size)

    bg = GameItem(os.path.join(GameConfig.root_path, GameConfig.resource_path, 'background.png'), GameConfig.window_size)

    grid = Grid((255, 255, 255), 1, 'line')
    h_points = [item for item in range(0, GameConfig.window_size[0], GameConfig.grid_size[0])]
    v_points = [item for item in range(0, GameConfig.window_size[1], GameConfig.grid_size[1])]
    h_lines = [((item, 0), (item, GameConfig.window_size[1])) for item in h_points]
    v_lines = [((0, item), (GameConfig.window_size[0], item)) for item in v_points]
    grid.containers += h_lines
    grid.containers += v_lines

    bricks = pygame.sprite.Group()
    for i, location_item, height_item in zip(range(4), [5, 5, 10, 10], [4, 3, 3, 4]):
        brick = Brick(os.path.join(GameConfig.root_path, GameConfig.resource_path, 'brick.png'), GameConfig.brick_size, GameConfig.grid_size)
        height = brick.image.get_size()[1] - height_item * GameConfig.grid_size[1]
        flip_flag = True
        location = [location_item * GameConfig.grid_size[0], 0]
        if i % 2 == 0:
            flip_flag = not flip_flag
            location[1] = GameConfig.window_size[1] - height_item * GameConfig.grid_size[1]
        brick.chop_with_flip(height, flip_flag, location)
        bricks.add(brick)

    birds = pygame.sprite.Group()
    for i in range(2):
        gender = 'male' if i % 2 == 0 else 'female'
        bird = Bird(os.path.join(GameConfig.root_path, GameConfig.resource_path, 'bird.png'), GameConfig.grid_size, gender, GameConfig.window_size)
        birds.add(bird)

    return game, bg, grid, bricks, birds


if __name__ == '__main__':
    game, *game_obj = game_env_init()
    game.loop(*game_obj, mode='value')
