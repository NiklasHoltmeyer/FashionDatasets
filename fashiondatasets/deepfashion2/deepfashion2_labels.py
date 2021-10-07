class DeepFashion2Labels:
    @staticmethod
    def categories():
        return [{'id': 1, 'name': 'short_sleeved_shirt', 'supercategory': 'clothes'},
                {'id': 2, 'name': 'long_sleeved_shirt', 'supercategory': 'clothes'},
                {'id': 3, 'name': 'short_sleeved_outwear', 'supercategory': 'clothes'},
                {'id': 4, 'name': 'long_sleeved_outwear', 'supercategory': 'clothes'},
                {'id': 5, 'name': 'vest', 'supercategory': 'clothes'},
                {'id': 6, 'name': 'sling', 'supercategory': 'clothes'},
                {'id': 7, 'name': 'shorts', 'supercategory': 'clothes'},
                {'id': 8, 'name': 'trousers', 'supercategory': 'clothes'},
                {'id': 9, 'name': 'skirt', 'supercategory': 'clothes'},
                {'id': 10, 'name': 'short_sleeved_dress', 'supercategory': 'clothes'},
                {'id': 11, 'name': 'long_sleeved_dress', 'supercategory': 'clothes'},
                {'id': 12, 'name': 'vest_dress', 'supercategory': 'clothes'},
                {'id': 13, 'name': 'sling_dress', 'supercategory': 'clothes'}]

    @staticmethod
    def category_id_to_name():
        return {1: 'short_sleeved_shirt',
                2: 'long_sleeved_shirt',
                3: 'short_sleeved_outwear',
                4: 'long_sleeved_outwear',
                5: 'vest',
                6: 'sling',
                7: 'shorts',
                8: 'trousers',
                9: 'skirt',
                10: 'short_sleeved_dress',
                11: 'long_sleeved_dress',
                12: 'vest_dress',
                13: 'sling_dress'}


