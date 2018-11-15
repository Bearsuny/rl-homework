class GameConfig:
    caption = 'lovebird-ff'
    window_size = (960, 540)
    grid_size = (60, 60)
    row = window_size[1] // grid_size[1]
    col = window_size[0] // grid_size[0]

    brick_size = (138, 793)
    bricks_pos = [5, 5, 10, 10]
    bricks_height = [4, 3, 3, 4]

    bird_size = (780, 690)
    female_bird_pos = col-1

    fps = 30
    save_number = 0
    loop_flag = False

    root_path = '.'
    resource_path = 'assets'
    save_path = 'output'


