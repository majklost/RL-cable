from pymunk.pygame_util import from_pygame
import pygame
import numpy as np

from ..objects.rectangle import Rectangle


class PMRectangleController:
    def __init__(self, rect: Rectangle, moving_force=100, color_change=True):
        self.rect = rect
        self.moving_force = moving_force
        self.color_change = color_change
        if self.color_change:
            rect.body.color = pygame.Color("blue")

    def update(self):
        force_template = np.zeros(4)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            force_template[0] = -self.moving_force
        if keys[pygame.K_RIGHT]:
            force_template[0] = self.moving_force
        if keys[pygame.K_UP]:
            force_template[1] = -self.moving_force
        if keys[pygame.K_DOWN]:
            force_template[1] = self.moving_force
        if keys[pygame.K_r]:
            self.rect.orientation += 0.1
        if keys[pygame.K_e]:
            self.rect.orientation -= 0.1
        # print(force_template)

        # self.rect.body.velocity = (force_template[:2]/200).tolist()
        self.rect.apply_force(force_template)
