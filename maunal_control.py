from gridworld import GridWorld
from window import Window


def redraw(img):
    window.show_img(img)


def reset():
    obs = env.reset()
    redraw(obs["image"])


def step(action):
    print(action)
    obs, reward, done, info = env.step(action)
    print('step=', env.step_count, ', reward=', reward, ',  done? ', done)

    if done:
        print('done!')
        reset()
    else:
        redraw(obs["image"])


def key_handler(event):
    if event.key == 'escape':
        window.close()
        return
    if event.key == 'backspace':
        reset()
        return
    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.up)
        return
    if event.key == 'down':
        step(env.actions.down)
        return


env = GridWorld(10, 10, True)
window = Window('Grid World')
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
