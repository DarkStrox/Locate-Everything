import unittest
from PIL import Image
from utils.parser import parse_grounding_response, DetectionBox
from utils.drawer import draw_annotations

class TestLocateEverythingParser(unittest.TestCase):
    def test_parse_standard_ref_box(self):
        text = "<ref>building</ref> <box> <0> <302> <695> <999> </box> <|im_end|>"
        dets = parse_grounding_response(text)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].label, "building")
        self.assertEqual(dets[0].ymin_norm, 0)
        self.assertEqual(dets[0].xmin_norm, 302)
        self.assertEqual(dets[0].ymax_norm, 695)
        self.assertEqual(dets[0].xmax_norm, 999)

    def test_parse_multiple_boxes_single_ref(self):
        text = "<ref>emirtare FLAg</ref><box><191><397><225><444></box><box><261><404><278><426></box><box><322><361><333><375></box>"
        dets = parse_grounding_response(text)
        self.assertEqual(len(dets), 3)
        for d in dets:
            self.assertEqual(d.label, "emirtare FLAg")
        self.assertEqual(dets[0].ymin_norm, 191)
        self.assertEqual(dets[1].ymin_norm, 261)
        self.assertEqual(dets[2].ymin_norm, 322)

    def test_parse_multiple_objects(self):
        text = "<ref>car</ref><box><100><200><300><400></box>\n<ref>person</ref><box><50><60><150><250></box>"
        dets = parse_grounding_response(text)
        self.assertEqual(len(dets), 2)
        self.assertEqual(dets[0].label, "car")
        self.assertEqual(dets[1].label, "person")

    def test_pixel_conversion(self):
        det = DetectionBox("tree", ymin_norm=0, xmin_norm=500, ymax_norm=1000, xmax_norm=1000)
        xmin, ymin, xmax, ymax = det.to_pixel_box(img_width=1000, img_height=500)
        self.assertEqual(xmin, 500)
        self.assertEqual(ymin, 0)
        self.assertEqual(xmax, 1000)
        self.assertEqual(ymax, 500)

    def test_drawer_output(self):
        img = Image.new("RGB", (600, 400), color=(100, 100, 100))
        dets = [DetectionBox("window", 50, 50, 300, 400)]
        annotated = draw_annotations(img, dets)
        self.assertIsNotNone(annotated)
        self.assertEqual(annotated.size, (600, 400))

if __name__ == "__main__":
    unittest.main()
