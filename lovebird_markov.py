import sys
import os

import pygame
from pygame.locals import *


class GameConfig:
    caption = 'lovebird-ff'
    window_size = (960, 540)
    grid_size = (60, 60)
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


class Background(GameItem):
    def __init__(self, source_path, size):
        super().__init__(source_path, size)

    def grid_blit(self, screen, size):
        screen_size = screen.get_size()
        h_points = [item for item in range(0, screen_size[0], size[0])]
        v_points = [item for item in range(0, screen_size[1], size[1])]
        h_lines = [((item, 0), (item, screen_size[1])) for item in h_points]
        v_lines = [((0, item), (screen_size[0], item)) for item in v_points]
        line_width = 1
        line_color = (255, 255, 255)
        for item in h_lines:
            pygame.draw.line(screen, line_color, *item, line_width)
        for item in v_lines:
            pygame.draw.line(screen, line_color, *item, line_width)


class LoveBirdGame():
    def __init__(self, caption, window_size):
        pygame.init()
        pygame.display.set_caption(caption)
        self.screen = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()

    def loop(self, bg_blit, grid_blit):
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

            bg_blit(self.screen)
            grid_blit(self.screen, GameConfig.grid_size)

            if GameConfig.loop_flag:
                pass

            pygame.display.update()


def dev():
    game = LoveBirdGame(GameConfig.caption, GameConfig.window_size)
    bg = Background(os.path.join(GameConfig.root_path, GameConfig.resource_path, 'background.png'), GameConfig.window_size)

    game.loop(bg.blit, bg.grid_blit)


def test():
    s = range(0, 960, 60)
    for item in s:
        print(item)


if __name__ == '__main__':
    dev()
    # test()
