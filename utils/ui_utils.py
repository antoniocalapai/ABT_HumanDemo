import pygame
import cv2
import numpy as np

display_w = pygame.display.Info().current_w

class ImageRect:
    def __init__(self,base_image_path,hover_image_path,select_image_path,width,height,pos):
        
        #image
        self.base_image = cv2.resize(cv2.imread(base_image_path, cv2.IMREAD_UNCHANGED), (width,height))
        self.hover_image = cv2.resize(cv2.imread(hover_image_path, cv2.IMREAD_UNCHANGED), (width,height))
        self.select_image = cv2.resize(cv2.imread(select_image_path, cv2.IMREAD_UNCHANGED), (width,height))
        self.width = width
        self.height = height
        self.pos = pos

    def draw(self, screen, status="base"):
        if status == "base":
            image = self.base_image
        elif status == "hover":
            image = self.hover_image
        elif status == "select":
            image = self.select_image

        if len(image.shape)==2:
            image = np.stack((image,image,image,image), axis=-1)
            mode = "BGRA"
        elif image.shape[2]==4:
            mode = "BGRA"
        elif image.shape[2]==3:
            mode = "BGR"
        screen.blit(pygame.image.frombuffer(image.tobytes(), (self.width,self.height), mode), self.pos)

    def get_rect(self):
        return [self.pos[0], self.pos[1], self.width, self.height]

class ImageButton:
    def __init__(self,image, hover_image ,width,height,pos,func, clickable=True, **args):
        #Core attributes 
        self.pressed = False

        #image
        # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (width,height))
        # hover_image = cv2.imread(hover_image_path, cv2.IMREAD_UNCHANGED)
        hover_image = cv2.resize(hover_image, (width,height))
        self.image = image
        self.hover_image = hover_image
        self.width = width
        self.height = height
        self.pos = pos
        self.clickable = clickable
        self.isClicked = 0

        # function
        self.func = func
        self.args = args

    def draw(self, screen):
        screen.blit(pygame.image.frombuffer(self.image.tobytes(), (self.image.shape[1],self.image.shape[0]), "RGBA"), self.pos)
        if self.clickable:
            self.check_click()
        if self.isClicked:
            screen.blit(pygame.image.frombuffer(self.hover_image.tobytes(), (self.width,self.height), "RGBA"), self.pos)

    def check_click(self):
        mouse_pos = pygame.mouse.get_pos()
        top_rect = pygame.Rect(self.pos,(self.width,self.height))
        if top_rect.collidepoint(mouse_pos):
            if pygame.mouse.get_pressed()[0]:
                self.pressed = True
            else:
                if self.pressed == True:
                    self.func(**self.args)
                    self.isClicked = True
                    self.pressed = False

class Button:
    def __init__(self,text,width,height,pos,func, clickable=True, color='#50938a', **args):
        #Core attributes 
        self.pressed = False
        self.original_y_pos = pos[1]
        self.clickable = clickable
        self.x = pos[0]
        self.y = pos[1]
        self.isClicked = 0

        # top rectangle 
        self.top_rect = pygame.Rect(pos,(width,height))
        self.color = color
        self.disable_color = '#798483'

        #text
        self.text_surf = pygame.font.Font(None,int(0.018*display_w)).render(text,True,'#ffffff')
        self.text_rect = self.text_surf.get_rect(center = self.top_rect.center)

        # function
        self.func = func
        self.args = args

    def draw(self, screen):
        # elevation logic 
        self.top_rect.y = self.original_y_pos
        self.text_rect.center = self.top_rect.center 

        pygame.draw.rect(screen, ('#80c7be' if self.isClicked else self.color) if self.clickable else self.disable_color , self.top_rect,border_radius = 6)
        screen.blit(self.text_surf, self.text_rect)
        if self.clickable:
            self.check_click()
            if self.isClicked >0:
                if self.isClicked ==1:
                    self.func(**self.args)
                self.isClicked-=1

    def check_click(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.top_rect.collidepoint(mouse_pos):
            # self.top_color = '#D74B4B'
            if pygame.mouse.get_pressed()[0]:
                self.pressed = True
            else:
                if self.pressed == True:
                    # self.func(**self.args)
                    self.isClicked = 5
                    self.pressed = False

class CheckBoxButton:
    def __init__(self,text,width,height,pos,func, clickable=True, active=False, color='#50938a', **args):
        #Core attributes 
        self.pressed = False
        self.original_y_pos = pos[1]
        self.clickable = clickable
        self.active = active

        # top rectangle 
        self.top_rect = pygame.Rect(pos,(width,height))
        self.active_color = color
        self.deactive_color = '#bfb760'
        self.disable_color = '#798483'

        #text
        self.text_surf = pygame.font.Font(None,32).render(text,True,'#ffffff')
        self.text_rect = self.text_surf.get_rect(center = self.top_rect.center)

        # function
        self.func = func
        self.args = args

    def draw(self, screen):
        # elevation logic 
        self.top_rect.y = self.original_y_pos
        self.text_rect.center = self.top_rect.center 

        pygame.draw.rect(screen, (self.active_color if self.active else self.deactive_color) if self.clickable else self.disable_color , self.top_rect,border_radius = 6)
        screen.blit(self.text_surf, self.text_rect)
        if self.clickable:
            self.check_click()

    def check_click(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.top_rect.collidepoint(mouse_pos):
            # self.top_color = '#D74B4B'
            if pygame.mouse.get_pressed()[0]:
                self.pressed = True
            else:
                if self.pressed == True:
                    self.active = not self.active
                    self.func(**self.args)
                    self.pressed = False

class Checkbox(pygame.sprite.Sprite):
    def __init__(self, surface, x, y, func=None, enable=True, default=False, caption="", **args):
        super().__init__()
        self.surface = surface
        # self.image = pygame.Surface([300,50])#, pygame.SRCALPHA, 32)
        self.image = pygame.Surface([0.19 * display_w, 0.025 * display_w])
        # self.image = self.image.convert_alpha()
        self.image.fill((255,255,255))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.x = x
        self.y = y
        self.caption = caption
        self.font = pygame.font.Font(None, int(0.014*display_w))
        # checkbox object
        self.checkbox_obj = pygame.Rect(0, 0, int(0.015*display_w), int(0.015*display_w))
        self.abs_rect = pygame.Rect(self.x, self.y, int(0.015*display_w), int(0.015*display_w))
        self.checkbox_outline = self.checkbox_obj.copy()
        # variables to test the different states of the checkbox
        self.default = default
        self.pressed = False
        self.checked = default
        self.enable = enable
        self.func = func
        self.args = args

    def _draw_button_text(self):
        font_surf = self.font.render(self.caption, True, (0, 0, 0) if self.enable else (100,100,100))
        w, h = self.font.size(self.caption)
        font_pos = (int(0.015*display_w) + 10, int(0.015*display_w) / 2 - h / 2)
        self.image.blit(font_surf, font_pos)

    def render_checkbox(self):
        self.surface.blit(self.image, (self.x, self.y))
        self.update()

    def update(self):
        if self.checked:
            pygame.draw.rect(self.image, (230, 230, 230), self.checkbox_obj)
            pygame.draw.rect(self.image, (0, 0, 0) if self.enable else (100,100,100), self.checkbox_outline, 1)
            pygame.draw.circle(self.image, '#50938a' if self.enable else '#709893', (int(0.015*display_w/2), int(0.015*display_w/2)), int(0.005*display_w))

        else:
            pygame.draw.rect(self.image, (230, 230, 230), self.checkbox_obj)
            pygame.draw.rect(self.image, (0, 0, 0) if self.enable else (100,100,100), self.checkbox_outline, 1)
        self._draw_button_text()

        if self.enable:
            mouse_pos = pygame.mouse.get_pos()
            if self.abs_rect.collidepoint(mouse_pos):
                if pygame.mouse.get_pressed()[0]:
                    self.pressed = True
                else:
                    if self.pressed == True:
                        self.checked = not self.checked
                        if self.func != None:
                            self.func(**self.args, state=self.checked)
                        self.pressed = False

    def click(self, state):
        self.checked = state
        if self.func != None:
            self.func(**self.args, state=self.checked)

    def reset(self):
        self.checked = self.default
            
class InputBox:
    def __init__(self, x, y, w, h, func=None, enable=True, text='', **args):
        self.FONT = pygame.font.Font(None, int(0.018*display_w))
        self.rect = pygame.Rect(x, y, w, h)
        self.color = (0,0,0)
        self.text = text
        self.txt_surface = self.FONT.render(text, True, self.color)
        self.active = False
        self.enable = enable
        self.func = func
        self.args = args
        self.cursor_blink_interval = 70
        self.cursor_blink = 0

    def handle_event(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                # If the user clicked on the input_box rect.
                if self.rect.collidepoint(event.pos):
                    # Toggle the active variable.
                    self.active = True
                    self.text = ''
                    if self.func:
                        self.func(**self.args)
                else:
                    self.active = False
                # Change the current color of the input box.
                self.color = (0,0,0) if self.active else (50,50,50)
            if event.type == pygame.KEYDOWN:
                if self.active:
                    # if event.key == pygame.K_RETURN:
                    #     if self.func:
                    #         self.func(**self.args)
                    if event.key == pygame.K_BACKSPACE:
                        self.text = self.text[:-1]
                    else:
                        unicode_char = event.unicode
                        if unicode_char and unicode_char.isprintable():
                            self.text += unicode_char
                    if self.func:
                        self.func(**self.args)
        
    def update(self):
        # Resize the box if the text is too long.
        width = max(200, self.FONT.render(self.text, True, self.color).get_width()+10)
        self.rect.w = width

    def draw(self, screen, events):
        # Blit the text.

        text = str(self.text)
        if self.active:
            if self.cursor_blink < self.cursor_blink_interval//2:
                text+='|'
            self.cursor_blink = (self.cursor_blink+1)%self.cursor_blink_interval
        pygame.draw.rect(screen, (230,230,230,230), self.rect ,border_radius = 6)
        t = self.FONT.render(text, True, self.color if self.enable else (100,100,100))
        screen.blit(t, (self.rect.x+10, self.rect.y+(self.rect.height - t.get_rect().height)//2))
        # Blit the rect.
        # pygame.draw.rect(screen, self.color, self.rect, 2)
        if self.enable:
            self.handle_event(events)
   
class Label:
    def __init__(self, x, y, w=display_w-200, text='', color=(0,0,0), pos='left'):
        self.FONT = pygame.font.Font(None, int(0.014*display_w))
        self.x = x
        self.y = y
        self.w = w
        self.default_text = text
        self.text = text
        self.color = color
        self.pos = pos

    def get_width(self):
        w = self.FONT.render(self.text, True, (0,0,0)).get_rect().width if isinstance(self.text, str) else 0
        return w
        
    def draw(self, screen):
        # Blit the text.
        y = self.y+5
        if isinstance(self.text, str):
            for line in self.text.split('\n'):
                text = self.FONT.render(line, True, self.color)
                if text.get_rect().width > self.w:
                    new_text = ''
                    while self.FONT.render(new_text, True, self.color).get_rect().width < self.w-10:
                        new_text = line[-1]+new_text
                        line=line[:-1]
                    new_text = '...'+new_text
                    text = self.FONT.render(new_text, True, self.color)

                if self.pos=='left':
                    screen.blit(text, (self.x+5, y))
                elif self.pos =='center':
                    screen.blit(text, (self.x+(self.w-text.get_rect().width)//2, y))
                h = text.get_rect().height
                y += h+10
        elif isinstance(self.text, list):
            for line in self.text:
                text = self.FONT.render(line, True, self.color)
                if text.get_rect().width > self.w:
                    new_text = ''
                    while self.FONT.render(new_text, True, self.color).get_rect().width < self.w-10:
                        new_text = line[-1]+new_text
                        line=line[:-1]
                    new_text = '...'+new_text
                    text = self.FONT.render(new_text, True, self.color)

                if self.pos=='left':
                    screen.blit(text, (self.x+5, y))
                elif self.pos =='center':
                    screen.blit(text, (self.x+(self.w-text.get_rect().width)//2, y))
                h = text.get_rect().height
                y += h+10

    def reset(self):
        self.text = self.default_text

class DropDown:
    def __init__(self, x, y, w, h, options, default=0, enable=True, func=None, scrollable=False, height=0):
        self.rect = pygame.Rect(x, y, w, h)
        self.font = pygame.font.Font(None, int(0.018*display_w))
        self.options = options
        self.draw_menu = False
        self.menu_active = False
        self.default = default
        self.selected_option = default
        self.active_option = -1
        self.enable = enable
        self.func = func
        self.scrollable = scrollable
        self.scroll_offset = 0
        self.SCREEN_HEIGHT = y+h+height
        self.rects = [pygame.Rect(self.rect.x, self.rect.y + i * (self.rect.height), self.rect.width, self.rect.height) for i in range(len(self.options))]

    def draw(self, surf):
        pygame.draw.rect(surf, '#50938a' if self.enable else '#798483', self.rect, 0, border_radius=6)
        msg = self.font.render(self.options[self.selected_option], 1, '#ffffff')
        surf.blit(msg, msg.get_rect(center = self.rect.center))

        if self.draw_menu:
            if self.scrollable:
                self.rects = [pygame.Rect(self.rect.x, self.rect.y + (i+1) * (self.rect.height), self.rect.width, self.rect.height) for i in range(len(self.options))]
                for i, text in enumerate(self.options):
                    offset_rect = self.rects[i].move(0, self.scroll_offset)
                    if offset_rect.y >= self.rect.bottom and offset_rect.bottom <= self.SCREEN_HEIGHT:
                        # rect = self.rect.copy()
                        # rect.y += (i+1) * self.rect.height
                        pygame.draw.rect(surf, '#50938a' if i == self.active_option else '#798483', offset_rect, 0)
                        msg = self.font.render(text, 1, '#ffffff')
                        surf.blit(msg, msg.get_rect(center = offset_rect.center))
            else:
                for i, text in enumerate(self.options):
                    rect = self.rect.copy()
                    rect.y += (i+1) * self.rect.height
                    pygame.draw.rect(surf, '#50938a' if i == self.active_option else '#798483', rect, 0)
                    msg = self.font.render(text, 1, '#ffffff')
                    surf.blit(msg, msg.get_rect(center = rect.center))

    def update(self, event_list):
        if not self.enable:
            return -1
        
        mpos = pygame.mouse.get_pos()
        self.menu_active = self.rect.collidepoint(mpos)
        
        self.active_option = -1
        if self.draw_menu:
            if self.scrollable:
                for i in range(len(self.options)):
                    offset_rect = self.rects[i].move(0, self.scroll_offset)
                    # rect = self.rect.copy()
                    # rect.y += (i+1) * self.rect.height
                    if offset_rect.collidepoint(mpos):
                        self.active_option = i
                        break
            else:
                for i in range(len(self.options)):
                    rect = self.rect.copy()
                    rect.y += (i+1) * self.rect.height
                    if rect.collidepoint(mpos):
                        self.active_option = i
                        break
                    
        max_offset = min(0, -(self.rect.bottom + len(self.rects) * (self.rect.height) - self.SCREEN_HEIGHT))
        for event in event_list:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.menu_active:
                        self.draw_menu = not self.draw_menu
                        self.scroll_offset = 0
                    elif self.draw_menu and self.active_option >= 0:
                        self.draw_menu = False
                        self.selected_option = self.active_option
                        if self.func != None:
                            self.func(self.selected_option)
                        return self.selected_option
                    else:
                        self.draw_menu = False
            if self.scrollable:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        self.scroll_offset = min(self.scroll_offset+5, 0)
                    elif event.button == 5:
                        self.scroll_offset = max(self.scroll_offset-5, max_offset)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        self.scroll_offset = max(self.scroll_offset-5, max_offset)
                    elif event.key == pygame.K_UP:
                        self.scroll_offset = min(self.scroll_offset+5, 0)


        return -1

    def get_active_option(self):
        return self.options[self.selected_option]

    def set_active_option(self, option):
        self.selected_option = option
    
    def reset(self):
        self.selected_option = self.default
    
class RadioButton:
    def __init__(self, surface, x, y, w, h, options, default=0, enable=True, dir='H', func=None, **args) -> None:
        self.surface = surface
        self.rect = pygame.Rect(x, y, w, h)
        self.rd = pygame.sprite.Group()
        self.buttons = []
        self.options = options
        self.func = func
        self.args = args
        for i, opt in enumerate(options):
            if dir=='H':
                self.buttons.append(Checkbox(surface, x+(w/len(options))*i, y, func=self.select, caption=opt, i=i))
            elif dir=='V':
                self.buttons.append(Checkbox(surface, x, y+(h)*i, func=self.select, caption=opt, i=i))
            self.rd.add(self.buttons[i])
        self.default = default
        self.selected_choice = default
        self.selected_option = options[default]
        self.enable = enable
        self.select(default)

    def set_enable(self, state=True):
        self.enable = state
        for i in range(len(self.buttons)):
            self.buttons[i].enable = state
        if self.enable:
            if self.func is not None:
                self.func(**self.args, selected=self.selected_option)

    def select(self, i, state=True):
        # if not self.enable:
        #     return
        for j,b in enumerate(self.buttons):
            self.buttons[j].checked=False
        self.buttons[i].checked=True
        self.selected_option = self.options[i]
        self.selected_choice = i
        if self.func is not None:
            if self.enable:
                self.func(**self.args, selected=self.options[i])

    def update(self):
        self.rd.update()

    def draw(self):
        self.rd.draw(self.surface)

    def reset(self):
        self.select(self.default)

class Toolkit:
    def __init__(self, surface, x, y, text='') -> None:
        icon = cv2.imread("utils/question_mark.png", -1)   
        self.w = int(0.015*display_w)
        self.icon = cv2.resize(icon, (self.w,self.w))
        self.x = x
        self.y = y
        self.surface = surface
        self.rect = pygame.Rect(x, y, self.w, self.w)
        self.lineSpacing = 5
        self.font = pygame.font.Font(None, int(0.012*display_w))
        self.fontHeight = self.font.size("Tg")[1]
        self.text = self.wrap_text(text)
        self.text_rect = pygame.Rect(0,0,310,len(self.text)*self.fontHeight+(len(self.text)+1)*self.lineSpacing)
        
    def wrap_text(self, text):
        lines = []
        while text:
            i = 1
            # determine maximum width of line
            while self.font.size(text[:i])[0] < 300 and i < len(text):
                i += 1
            # if we've wrapped the text, then adjust the wrap to the last word      
            if i < len(text): 
                i = text.rfind(" ", 0, i) + 1
            lines.append(text[:i])
            text = text[i:]

        return lines
    
    def draw(self):
        self.surface.blit(pygame.image.frombuffer(self.icon.tobytes(), (self.w,self.w), "RGBA"), (self.x,self.y))
        self.update()

    def update(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            self.text_rect.x = mouse_pos[0]+10
            self.text_rect.y = mouse_pos[1]+10
            pygame.draw.rect(self.surface, (210,210,210), self.text_rect)
            for i, t in enumerate(self.text):
                self.surface.blit(self.font.render(t, True, (0,0,0)), (self.text_rect.x+5, self.text_rect.y+self.lineSpacing+i*(self.lineSpacing+self.fontHeight)))

class List:
    def __init__(self, x, y, w, h, height, options, enable=True, func=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.height = height
        self.font = pygame.font.Font(None, int(0.016*display_w))
        self.options = options
        self.status = [False]*len(options)
        self.draw_menu = False
        self.menu_active = False
        self.active_option = -1
        self.enable = enable
        self.func = func
        self.selected = set()
        self.scroll_offset = 0
        self.rects = [pygame.Rect(self.rect.x, self.rect.y + i * (self.rect.height), self.rect.width, self.rect.height) for i in range(len(self.options))]
        self.SCREEN_HEIGHT = y+self.height
        self.screen_rect = pygame.Rect(self.rect.x-10, self.rect.y-10, self.rect.width+20, self.height+20)

    def draw(self, surf):
        pygame.draw.rect(surf, '#221133', self.screen_rect, 1)
        self.rects = [pygame.Rect(self.rect.x, self.rect.y + i * (self.rect.height), self.rect.width, self.rect.height) for i in range(len(self.options))]
        for i, text in enumerate(self.options):
            # rect = self.rect.copy()
            # rect.y += (i) * self.rect.height
            offset_rect = self.rects[i].move(0, self.scroll_offset)
            if offset_rect.y >= self.rect.y and offset_rect.bottom <= self.SCREEN_HEIGHT:
                pygame.draw.rect(surf, '#666666' if i in self.selected else '#eeeeee', offset_rect, 0)
                msg = self.font.render(text, 1, '#eeeeee' if i in self.selected else '#111111')
                surf.blit(msg, msg.get_rect(center = offset_rect.center))

    def update(self,events):
        if not self.enable:
            return -1
        
        max_offset = min(0, -(len(self.rects) * (self.rect.height + 10) - self.SCREEN_HEIGHT + 20))
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if not self.screen_rect.collidepoint(event.pos):
                    continue
                if event.button == 4:  # Mouse wheel up
                    self.scroll_offset = min(self.scroll_offset+5, 0)
                elif event.button == 5:  # Mouse wheel down
                    self.scroll_offset = max(self.scroll_offset-5, max_offset)
                elif event.button == 1:
                    for i in range(len(self.options)):
                        offset_rect = self.rects[i].move(0, self.scroll_offset)
                        # rect = self.rect.copy()
                        # rect.y += (i) * self.rect.height
                        if offset_rect.collidepoint(event.pos):
                            if i in self.selected:
                                self.selected.remove(i)
                            else:
                                self.selected.add(i)
                            self.func(self.options[i], i in self.selected)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    self.scroll_offset = max(self.scroll_offset-5, max_offset)
                elif event.key == pygame.K_UP:
                    self.scroll_offset = min(self.scroll_offset+5, 0)

    def select_all(self, status):
        if status:
            self.selected = set(list(range(len(self.options))))
        else:
            self.selected = set()

    def reset(self):
        self.selected = set()