import sys
import numpy as np
import pygame
from pygame.locals import *


def screen_blit(screen, bg, birds, bricks, window_size):
    screen.blit(bg, (0, 0))
    for item in birds:
        screen.blit(item.image, item.rect)
    for item in bricks:
        screen.blit(item.image, item.rect)


class Bird(pygame.sprite.Sprite):
    def __init__(self, file_path, bird_height, gender, window_size=None):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(file_path).convert_alpha()
        self.image = pygame.transform.scale(self.image, (self.image.get_size()[0] * bird_height // self.image.get_size()[1], bird_height))
        self.rect = self.image.get_rect()
        if gender == 'female' and window_size != None:
            self.image = pygame.transform.flip(self.image, True, False).convert_alpha()
            self.rect[0] = window_size[0]-self.image.get_size()[0]
        self.image = self.image.convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)

    def update(self, step, direction):
        if direction == 0: # Down
            self.rect.move_ip(0, step)
        if direction == 1: # Right
            self.rect.move_ip(step, 0)
        if direction == 2: # Up
            self.rect.move_ip(0, -step)
        if direction == 3: # Left
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


class Brick(pygame.sprite.Sprite):
    def __init__(self, file_path, brick_weight):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(file_path).convert_alpha()
        self.image = pygame.transform.scale(self.image, (brick_weight, self.image.get_size()[1]*brick_weight // self.image.get_size()[0]))
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)


def bg_init(file_path, window_size):
    bg = pygame.image.load(file_path).convert_alpha()
    bg = pygame.transform.scale(bg, (bg.get_size()[1] * window_size[0] // window_size[1], window_size[1]))
    return bg


def bricks_init(file_path, brick_weight, chop_heights):
    bricks = pygame.sprite.Group()
    for i, item in enumerate(chop_heights):
        brick = Brick(file_path, brick_weight=100)
        brick.image = pygame.transform.chop(brick.image, [0, brick.image.get_size()[1] - item, 0, item])
        if i % 2 == 0:
            brick.image = pygame.transform.flip(brick.image, False, True)
        location = [0, 0]
        window_middle = window_size[0] // 2
        intervel = 150
        location[0] = window_middle - intervel - brick.image.get_size()[0] if i < 2 else window_middle + intervel
        location[1] = 0 if i % 2 == 0 else window_size[1] - brick.image.get_size()[1]
        brick.rect[0], brick.rect[1] = location[0], location[1]
        brick.image = brick.image.convert_alpha()
        brick.mask = pygame.mask.from_surface(brick.image)
        bricks.add(brick)
    return bricks

def find_mate(bird_male, bird_female):
    return True if pygame.sprite.collide_mask(bird_male, bird_female) else False


if __name__ == '__main__':
    np.random.seed(0)

    pygame.init()
    pygame.display.set_caption('lovebirds-fengfan')

    clock = pygame.time.Clock()

    window_size = (1066, 600)
    screen = pygame.display.set_mode(window_size)

    bg = bg_init('./assets/bg.png', window_size)
    bird_male = Bird('./assets/bird.png', bird_height=100, gender='male')
    bird_female = Bird('./assets/bird.png', bird_height=100, gender='female', window_size=window_size)
    birds = [bird_male, bird_female]
    bricks = bricks_init('./assets/brick.png', brick_weight=100, chop_heights=[275, 425, 375, 325])

    start_flag = False
    save_screen_no = 0
    while True:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_RETURN:
                    start_flag = not start_flag
                if event.key == K_ESCAPE:
                    pygame.image.save(screen, f'screen_{save_screen_no}.png')
                    save_screen_no += 1
            elif event.type == QUIT:
                pygame.quit()
                sys.exit()
        screen_blit(screen, bg, birds, bricks, window_size)
        pygame.display.update()
        if start_flag:
            step = 50
            for bird in birds:
                direction = bird.collision(step, bricks, window_size)
                bird.update(step, direction)
            if find_mate(bird_male, bird_female):
                start_flag = False
