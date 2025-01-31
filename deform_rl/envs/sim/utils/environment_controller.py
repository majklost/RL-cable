# Similar to cable controller but reacts to actions instead
import pygame
import numpy as np


class EnvironmentController:
    def __init__(self, num_segments):
        self.current = 0
        self.num_segments = num_segments

    def _cur_next(self):
        self.current = (self.current + 1) % self.num_segments

    def _cur_prev(self):
        self.current = (self.current - 1) % self.num_segments

    def get_action(self):
        all_actions = np.zeros((self.num_segments, 2))
        force_template = self._build_force()
        all_actions[self.current] = force_template
        return all_actions.flatten()

    def _build_force(self):
        force_template = np.zeros(2)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB and event.mod & pygame.KMOD_SHIFT:
                    print("shift tab")
                    self._cur_prev()
                elif event.key == pygame.K_TAB:
                    self._cur_next()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            force_template[0] = -1
        if keys[pygame.K_RIGHT]:
            force_template[0] = 1
        if keys[pygame.K_UP]:
            force_template[1] = -1
        if keys[pygame.K_DOWN]:
            force_template[1] = 1
        return force_template
    
class LinearController:
    pass
