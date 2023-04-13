import dominate
from dominate.tags import *
import os


class HTML:
    def __init__(self, web_dir, page_title, reflesh=0):
        self.page_title = page_title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        self.t = None
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        # print(self.img_dir)

        self.doc = dominate.document(title=page_title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, string):
        with self.doc:
            h3(string)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=400, height=0):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, image_link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', image_link)):
                                if height != 0:
                                    img(style="width:%dpx;height:%dpx" % (width, height), src=os.path.join('images',
                                                                                                           im))
                                else:
                                    img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()
