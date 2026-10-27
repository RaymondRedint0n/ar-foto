"""Собирает сайт AR-фото: вписывает сегодняшнюю дату в табличку и собирает standee.glb.

Запуск:  python build_date.py                 -> папка _site с index.html и standee.glb
         DATE=2026-12-31 python build_date.py  -> собрать на конкретную дату (для проверки)
"""
import json, os, shutil, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np, trimesh
from PIL import Image, ImageDraw, ImageFont
from trimesh.visual.material import PBRMaterial
from trimesh.visual import TextureVisuals

TZ = ZoneInfo('Asia/Yekaterinburg')      # дата меняется в полночь по Екатеринбургу
GAP = 0.004                              # половина толщины «картона», м
ROOT = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(ROOT, 'assets')         # здесь же лежат шрифты
OUT = os.path.join(ROOT, '_site')

# Девушки календаря. Новая девушка = папка в assets (front.png, back.png, sign.json) + строка здесь.
#   assets — папка с картинками, out — папка на сайте ('' = главная страница),
#   height — высота силуэта от верхней точки (макушка или поднятая рука) до подошвы, м
FIGURES = [
    {'assets': 'assets',          'out': '',         'height': 1.70},
    {'assets': 'assets/balerina', 'out': 'balerina', 'height': 1.98},   # рука поднята над головой, на пуантах
]

MONTHS = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля',
          'августа', 'сентября', 'октября', 'ноября', 'декабря']
WEEKDAYS = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']
ORANGE = (242, 107, 20); INK = (24, 24, 26); GREY = (120, 120, 124)


def today():
    if os.environ.get('DATE'):
        return dt.date.fromisoformat(os.environ['DATE'])
    return dt.datetime.now(TZ).date()


def font(bold, size):
    return ImageFont.truetype(os.path.join(A, 'font-bold.ttf' if bold else 'font.ttf'), size)


def fit(draw, text, bold, max_w, size):
    """Подбирает кегль, чтобы строка влезла по ширине."""
    while size > 10:
        f = font(bold, size)
        if draw.textlength(text, font=f) <= max_w:
            return f
        size -= 2
    return font(bold, size)


def sign_card(d, W=600, H=830):
    """Макет таблички: белый лист, тёмный текст, оранжевый акцент."""
    im = Image.new('RGB', (W, H), 'white'); g = ImageDraw.Draw(im)
    pad = 46; mw = W - 2 * pad

    def center(text, f, y, fill, spacing=0):
        if spacing:
            w = sum(g.textlength(ch, font=f) for ch in text) + spacing * (len(text) - 1)
            x = (W - w) / 2
            for ch in text:
                g.text((x, y), ch, font=f, fill=fill); x += g.textlength(ch, font=f) + spacing
        else:
            g.text(((W - g.textlength(text, font=f)) / 2, y), text, font=f, fill=fill)

    f_top = font(True, 40)
    center('СЕГОДНЯ', f_top, 52, GREY, spacing=10)
    g.rectangle([W / 2 - 60, 116, W / 2 + 60, 124], fill=ORANGE)

    day = str(d.day)
    f_day = fit(g, day, True, mw, 360)
    bb = g.textbbox((0, 0), day, font=f_day)
    g.text(((W - (bb[2] - bb[0])) / 2 - bb[0], 150 - bb[1]), day, font=f_day, fill=INK)
    y = 150 + (bb[3] - bb[1]) + 40

    month = MONTHS[d.month - 1].upper()
    f_m = fit(g, month, True, mw, 92)
    center(month, f_m, y, ORANGE); y += f_m.size + 34

    g.line([pad + 40, y, W - pad - 40, y], fill=(205, 205, 208), width=4); y += 30
    wd = WEEKDAYS[d.weekday()]
    f_w = fit(g, wd, False, mw, 66)
    center(wd, f_w, y, INK); y += f_w.size + 22
    center(str(d.year), font(False, 40), y, GREY)
    return im


def perspective_coeffs(dst, src):
    """Коэффициенты PIL PERSPECTIVE: точка результата -> точка исходника."""
    M = []; B = []
    for (x, y), (u, v) in zip(dst, src):
        M.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        M.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    return np.linalg.solve(np.array(M, float), np.array(B, float)).tolist()


def front_with_date(d, meta, folder=A):
    front = Image.open(os.path.join(folder, 'front.png')).convert('RGBA')
    card = sign_card(d)
    q = meta['front_sign_inner']
    src = [(0, 0), (card.width, 0), (card.width, card.height), (0, card.height)]
    warped = card.transform(front.size, Image.PERSPECTIVE, perspective_coeffs(q, src), Image.BICUBIC, fillcolor='white')
    mask = Image.new('L', front.size, 0); ImageDraw.Draw(mask).polygon([tuple(p) for p in q], fill=255)
    rgb = np.array(front.convert('RGB')).astype(float)
    ink = np.array(warped).astype(float) / 255
    m = (np.array(mask) > 0)[..., None]
    out = np.where(m, rgb * ink, rgb)        # «умножение»: свет и тень на листе сохраняются
    res = Image.fromarray(out.clip(0, 255).astype('uint8')); res.putalpha(front.split()[3])
    return res


def silhouette_info(im):
    a = np.array(im)[:, :, 3] > 128
    ys, xs = np.nonzero(a)
    top, bot = ys.min(), ys.max()
    feet_x = np.nonzero(a[bot - 240:bot + 1])[1].mean()
    return top, bot, feet_x


def quad(im, S, front):
    w, h = im.size
    top, bot, fx = silhouette_info(im)
    x0 = -fx * S; x1 = (w - fx) * S
    y0 = -(h - 1 - bot) * S; y1 = y0 + h * S
    z = GAP if front else -GAP
    if front:
        v = [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]
    else:   # вид сзади: левый край картинки — слева у зрителя за спиной фигуры
        v = [[-x0, y0, z], [-x1, y0, z], [-x1, y1, z], [-x0, y1, z]]
    f = [[0, 1, 2], [0, 2, 3]]
    uv = [[0, 0], [1, 0], [1, 1], [0, 1]]
    mat = PBRMaterial(baseColorTexture=im, metallicFactor=0.0, roughnessFactor=1.0,
                      alphaMode='MASK', alphaCutoff=0.5, doubleSided=False,
                      emissiveTexture=im.convert('RGB'), emissiveFactor=[0.22] * 3)
    return trimesh.Trimesh(np.array(v, float), f, visual=TextureVisuals(uv=np.array(uv, float), material=mat), process=False)


def build_figure(d, cfg):
    folder = os.path.join(ROOT, cfg['assets'])
    out = os.path.join(OUT, cfg['out'])
    meta = json.load(open(os.path.join(folder, 'sign.json')))
    front = front_with_date(d, meta, folder)
    back = Image.open(os.path.join(folder, 'back.png')).convert('RGBA')
    t, b, _ = silhouette_info(front)
    S = cfg['height'] / (b - t)
    sc = trimesh.Scene()
    sc.add_geometry(quad(front, S, True), node_name='front', geom_name='front')
    sc.add_geometry(quad(back, S, False), node_name='back', geom_name='back')
    os.makedirs(out, exist_ok=True)
    sc.export(os.path.join(out, 'standee.glb'))
    # картинки для режима «Фото с фигурой» (камера телефона + фигура поверх)
    for im, name in [(front, 'figure.png'), (back, 'figure-back.png')]:
        im = im.crop(im.getbbox())
        if im.height > 1600:
            im = im.resize((round(im.width * 1600 / im.height), 1600), Image.LANCZOS)
        im.save(os.path.join(out, name), optimize=True)
    shutil.copy(os.path.join(ROOT, 'index.html'), os.path.join(out, 'index.html'))
    for extra in ('live.mp4', 'live.json'):   # «живое фото», если есть
        if os.path.exists(os.path.join(folder, extra)):
            shutil.copy(os.path.join(folder, extra), os.path.join(out, extra))
    print('  ', cfg['out'] or '(главная)', '->', out)


def main():
    d = today()
    print('built for', d.isoformat())
    for cfg in FIGURES:
        build_figure(d, cfg)
    with open(os.path.join(ROOT, 'last_build.txt'), 'w') as fh:
        fh.write(d.isoformat() + '\n')


if __name__ == '__main__':
    main()
