import os
import sys


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

import cv2
import numpy as np
from best_fit import fit
from rectangle import Rectangle
from note import Note

staff_files = [
    "resources/template/staff2.png",
    "resources/template/staff.png"]
quarter_files = [
    "resources/template/quarter.png",
    "resources/template/solid-note.png"]
sharp_files = [
    "resources/template/sharp.png"]
flat_files = [
    "resources/template/flat-line.png",
    "resources/template/flat-space.png"]
half_files = [
    "resources/template/half-space.png",
    "resources/template/half-note-line.png",
    "resources/template/half-line.png",
    "resources/template/half-note-space.png"]
whole_files = [
    "resources/template/whole-space.png",
    "resources/template/whole-note-line.png",
    "resources/template/whole-line.png",
    "resources/template/whole-note-space.png"]


staff_imgs = [cv2.imread(staff_file, 0) for staff_file in staff_files]
quarter_imgs = [cv2.imread(quarter_file, 0) for quarter_file in quarter_files]
sharp_imgs = [cv2.imread(sharp_files, 0) for sharp_files in sharp_files]
flat_imgs = [cv2.imread(flat_file, 0) for flat_file in flat_files]
half_imgs = [cv2.imread(half_file, 0) for half_file in half_files]
whole_imgs = [cv2.imread(whole_file, 0) for whole_file in whole_files]


kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
kernel2 = np.array([[-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1]])


def locate_images(img, templates, start, stop, threshold):
    locations, scale = fit(img, templates, start, stop, threshold)
    img_locations = []
    for i in range(len(templates)):
        w, h = templates[i].shape[::-1]
        w *= scale
        h *= scale
        try:
            img_locations.append([Rectangle(pt[0], pt[1], w, h) for pt in zip(*locations[i][::-1])])
        except IndexError:
            pass
    return img_locations


def merge_recs(recs, threshold):
    filtered_recs = []
    while len(recs) > 0:
        r = recs.pop(0)
        recs.sort(key=lambda rec: rec.distance(r))
        merged = True
        while (merged):
            merged = False
            i = 0
            for _ in range(len(recs)):
                if r.overlap(recs[i]) > threshold or recs[i].overlap(r) > threshold:
                    r = r.merge(recs.pop(i))
                    merged = True
                elif recs[i].distance(r) > r.w / 2 + recs[i].w / 2:
                    break
                else:
                    i += 1
        filtered_recs.append(r)
    return filtered_recs


def SheetVision(img_file):
    img = cv2.imread(img_file, 0)
    img_gray0 = img  # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.filter2D(img_gray0, -1, kernel)
    img_staff = cv2.filter2D(img_gray0, -1, kernel2)
    img_staff = cv2.bitwise_not(img_staff)
    # cv2.imwrite('staff_recs_img.png', img_staff)
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    ret, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    img_width, img_height = img_gray.shape[::-1]


    name = os.path.basename(img_file)
    name = os.path.splitext(name)[0]

    staff_lower, staff_upper, staff_thresh = 50, 150, 0.5 # * (np.sqrt(img_width * img_height) / 1000)
    sharp_lower, sharp_upper, sharp_thresh = 50, 150, 0.6  # * (np.sqrt(img_width * img_height) / 1000)
    flat_lower, flat_upper, flat_thresh = 50, 150, 0.65  # * (np.sqrt(img_width * img_height) / 1000)
    quarter_lower, quarter_upper, quarter_thresh = 50, 150, 0.4  # * (np.sqrt(img_width * img_height) / 1000)
    half_lower, half_upper, half_thresh = 50, 150, 0.55  # * (np.sqrt(img_width * img_height) / 1000)
    whole_lower, whole_upper, whole_thresh = 50, 150, 0.6  # * (np.sqrt(img_width * img_height) / 1000)

    # print("Matching staff image...")
    staff_recs = locate_images(img_staff, staff_imgs, staff_lower, staff_upper, staff_thresh)

    # print("Filtering weak staff matches...")
    staff_recs = [j for i in staff_recs for j in i]
    heights = [r.y for r in staff_recs] + [0]
    histo = [heights.count(i) for i in range(0, max(heights) + 1)]
    avg = np.mean(list(set(histo)))
    staff_recs = [r for r in staff_recs if histo[r.y] > avg]

    # print("Merging staff image results...")
    staff_recs = merge_recs(staff_recs, 0.01)
    staff_recs_img = img.copy()
    for r in staff_recs:
        r.draw(staff_recs_img, (0, 0, 255), 2)
    cv2.imwrite(f'staff_recs_img{name}.png', staff_recs_img)
    # open_file('staff_recs_img.png')

    # print("Discovering staff locations...")
    staff_boxes = merge_recs([Rectangle(0, r.y, img_width, r.h) for r in staff_recs], 0.01)
    staff_boxes_img = img.copy()
    for r in staff_boxes:
        r.draw(staff_boxes_img, (0, 0, 255), 2)
    cv2.imwrite(f'staff_boxes_img{name}.png', staff_boxes_img)
    # open_file('staff_boxes_img.png')

    # print("Matching sharp image...")
    sharp_recs = locate_images(img_gray, sharp_imgs, sharp_lower, sharp_upper, sharp_thresh)

    # print("Merging sharp image results...")
    sharp_recs = merge_recs([j for i in sharp_recs for j in i], 0.5)
    sharp_recs_img = img.copy()
    for r in sharp_recs:
        r.draw(sharp_recs_img, (0, 0, 255), 2)
    cv2.imwrite(f'sharp_recs_img{name}.png', sharp_recs_img)
    # open_file('sharp_recs_img.png')

    # print("Matching flat image...")
    flat_recs = locate_images(img_gray, flat_imgs, flat_lower, flat_upper, flat_thresh)

    # print("Merging flat image results...")
    flat_recs = merge_recs([j for i in flat_recs for j in i], 0.5)
    flat_recs_img = img.copy()
    for r in flat_recs:
        r.draw(flat_recs_img, (0, 0, 255), 2)
    cv2.imwrite(f'flat_recs_img{name}.png', flat_recs_img)
    # open_file('flat_recs_img.png')

    # print("Matching quarter image...")
    quarter_recs = locate_images(img_gray, quarter_imgs, quarter_lower, quarter_upper, quarter_thresh)

    # print("Merging quarter image results...")
    quarter_recs = merge_recs([j for i in quarter_recs for j in i], 0.5)
    quarter_recs_img = img.copy()
    for r in quarter_recs:
        r.draw(quarter_recs_img, (0, 0, 255), 2)
    cv2.imwrite(f'quarter_recs_img{name}.png', quarter_recs_img)
    # open_file('quarter_recs_img.png')

    # print("Matching half image...")
    half_recs = locate_images(img_gray, half_imgs, half_lower, half_upper, half_thresh)

    # print("Merging half image results...")
    half_recs = merge_recs([j for i in half_recs for j in i], 0.5)
    half_recs_img = img.copy()
    for r in half_recs:
        r.draw(half_recs_img, (0, 0, 255), 2)
    cv2.imwrite(f'half_recs_img{name}.png', half_recs_img)
    # open_file('half_recs_img.png')

    # print("Matching whole image...")
    whole_recs = locate_images(img_gray, whole_imgs, whole_lower, whole_upper, whole_thresh)

    # print("Merging whole image results...")
    whole_recs = merge_recs([j for i in whole_recs for j in i], 0.5)
    whole_recs_img = img.copy()
    for r in whole_recs:
        r.draw(whole_recs_img, (0, 0, 255), 2)
    cv2.imwrite(f'whole_recs_img{name}.png', whole_recs_img)
    # open_file('whole_recs_img.png')

    note_groups = []
    prev_note = None
    no_num = lambda x: "" if x[-1] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] else x[-1]
    with open(f"{name}.txt", "w") as f:
        for b, box in enumerate(staff_boxes):
            staff_sharps = [Note(r, "sharp", box)
                            for r in sharp_recs if abs(r.middle[1] - box.middle[1]) < box.h * 5.0 / 8.0]
            staff_flats = [Note(r, "flat", box)
                           for r in flat_recs if abs(r.middle[1] - box.middle[1]) < box.h * 5.0 / 8.0]
            quarter_notes = [Note(r, "4,8", box, staff_sharps, staff_flats)
                             for r in quarter_recs if abs(r.middle[1] - box.middle[1]) < box.h * 5.0 / 8.0]
            half_notes = [Note(r, "2", box, staff_sharps, staff_flats)
                          for r in half_recs if abs(r.middle[1] - box.middle[1]) < box.h * 5.0 / 8.0]
            whole_notes = [Note(r, "1", box, staff_sharps, staff_flats)
                           for r in whole_recs if abs(r.middle[1] - box.middle[1]) < box.h * 5.0 / 8.0]
            staff_notes = quarter_notes + half_notes + whole_notes
            if len(staff_notes) == 0:
                continue
            if b + 1 < len(staff_boxes) and staff_boxes[b + 1].distance(box) -(staff_boxes[b + 1].h + box.h)/2 <= (staff_boxes[b + 1].h + box.h)/6:
                staff_sharps = [Note(r, "sharp", staff_boxes[b + 1])
                                for r in sharp_recs if abs(r.middle[1] - staff_boxes[b + 1].middle[1]) < staff_boxes[b + 1].h * 5.0 / 8.0]
                staff_flats = [Note(r, "flat", staff_boxes[b + 1])
                               for r in flat_recs if abs(r.middle[1] - staff_boxes[b + 1].middle[1]) < staff_boxes[b + 1].h * 5.0 / 8.0]
                quarter_notes = [Note(r, "4,8", staff_boxes[b + 1], staff_sharps, staff_flats, unseen="f_clef" )
                                 for r in quarter_recs if abs(r.middle[1] - staff_boxes[b + 1].middle[1]) < staff_boxes[b + 1].h * 5.0 / 8.0]
                half_notes = [Note(r, "2", staff_boxes[b + 1], staff_sharps, staff_flats, unseen="f_clef")
                              for r in half_recs if abs(r.middle[1] - staff_boxes[b + 1].middle[1]) < staff_boxes[b + 1].h * 5.0 / 8.0]
                whole_notes = [Note(r, "1", staff_boxes[b + 1], staff_sharps, staff_flats, unseen="f_clef")
                               for r in whole_recs if abs(r.middle[1] - staff_boxes[b + 1].middle[1]) < staff_boxes[b + 1].h * 5.0 / 8.0]
                temp_len = len(staff_notes)
                staff_notes = staff_notes + quarter_notes + half_notes + whole_notes
                if len(staff_notes) == temp_len:
                    staff_notes.sort(key=lambda n: n.rec.x)
                    for note in staff_notes:
                        if prev_note and abs(prev_note.rec.x - note.rec.x) < 10:
                            # temp = img_gray.copy()
                            # cv2.circle(temp, (note.rec.x, note.rec.y), 20, (0, 255, 0), 5)
                            # cv2.imshow(f"{note.note}", temp)
                            # cv2.waitKey(0)
                            f.write(" " + f"{note.note[0]}")#{no_num(note.note[-1])}@({note.rec.x},{note.rec.y})")
                        else:
                            # temp = img_gray.copy()
                            # cv2.circle(temp, (note.rec.x, note.rec.y), 20, (0, 255, 0), 5)
                            # cv2.imshow(f"{note.note}", temp)
                            # cv2.waitKey(0)
                            f.write("\n" + f"{note.note[0]}")#{no_num(note.note[-1])}@({note.rec.x},{note.rec.y})")
                        prev_note = note
                staff_notes.sort(key=lambda n: n.rec.x)
                for note in staff_notes:
                    if prev_note and abs(prev_note.rec.x - note.rec.x) < 10:
                        # temp = img_gray.copy()
                        # cv2.circle(temp, (note.rec.x, note.rec.y), 20, (0, 255, 0), 5)
                        # cv2.imshow(f"{note.note}", temp)
                        # cv2.waitKey(0)
                        f.write(" "  + f"{note.note[0]}")#{no_num(note.note[-1])}@({note.rec.x},{note.rec.y})")
                    else:
                        # temp = img_gray.copy()
                        # cv2.circle(temp, (note.rec.x, note.rec.y), 20, (0, 255, 0), 5)
                        # cv2.imshow(f"{note.note}", temp)
                        # cv2.waitKey(0)
                        f.write("\n"  + f"{note.note[0]}")#{no_num(note.note[-1])}@({note.rec.x},{note.rec.y})")
                    prev_note = note
                continue
            else:
                staff_notes.sort(key=lambda n: n.rec.x)
                for note in staff_notes:
                    if prev_note and abs(prev_note.rec.x - note.rec.x) < 10:
                        # temp = img_gray.copy()
                        # cv2.circle(temp, (note.rec.x, note.rec.y), 20, (0, 255, 0), 5)
                        # cv2.imshow(f"{note.note}", temp)
                        # cv2.waitKey(0)
                        f.write(" " + f"{note.note[0]}")#{no_num(note.note[-1])}@({note.rec.x},{note.rec.y})")
                    else:
                        # temp = img_gray.copy()
                        # cv2.circle(temp, (note.rec.x, note.rec.y), 20, (0, 255, 0), 5)
                        # cv2.imshow(f"{note.note}", temp)
                        # cv2.waitKey(0)
                        f.write("\n"  + f"{note.note[0]}")#{no_num(note.note[-1])}@({note.rec.x},{note.rec.y})")
                    prev_note = note

    #         staffs = [r for r in staff_recs if r.overlap(box) > 0]
    #         staffs.sort(key=lambda r: r.x)
    #         for staff in staffs:
    #             print(staff.x, staff.y, staff.w, staff.h)
    #         for note in staff_notes:
    #             print(note.note, note.rec.x, note.rec.y)
    #         note_group = []
    #         i = 0;
    #         j = 0;
    #         while (i < len(staff_notes)):
    #             if (j < len(staffs) and staff_notes[i].rec.x > staffs[j].x):
    #                 r = staffs[j]
    #                 j += 1;
    #                 if len(note_group) > 0:
    #                     note_groups.append(note_group)
    #                     note_group = []
    #                 note_color = (randint(0, 255), randint(0, 255), randint(0, 255))
    #             else:
    #                 note_group.append(staff_notes[i])
    #                 staff_notes[i].rec.draw(img, note_color, 2)
    #                 i += 1
    #         note_groups.append(note_group)
    #
    #     # for r in staff_boxes:
    #     #     r.draw(img, (0, 0, 255), 2)
    #     # for r in sharp_recs:
    #     #     r.draw(img, (0, 0, 255), 2)
    #     # flat_recs_img = img.copy()
    #     # for r in flat_recs:
    #     #     r.draw(img, (0, 0, 255), 2)
    #
    #     # cv2.imwrite('res.png', img)
    #     # open_file('res.png')
    # name = os.path.basename(img_file)
    # name = os.path.splitext(name)[0]
    # with open(f"{name}.txt", "w") as f:
    #     for note_group in note_groups:
    #         notes = [note.note for note in note_group]
    #         for note in notes:
    #             f.write(note)
    #             f.write("\n")

    print(f"done page")
