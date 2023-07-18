import os

def convert_xml_file_to_txt(xml_file_path, output_path,names):
    with open(xml_file_path, 'r') as xml_file:
        xml_content = xml_file.read()

    # Find all occurrences of <object> tag
    object_start = '<object>'
    object_end = '</object>'
    object_indices = []
    start_index = xml_content.find(object_start)
    while start_index != -1:
        end_index = xml_content.find(object_end, start_index)
        if end_index != -1:
            object_indices.append((start_index, end_index + len(object_end)))
        start_index = xml_content.find(object_start, end_index)

    labels = []
    for start, end in object_indices:
        object_xml = xml_content[start:end]
        values = extract_object_values(object_xml)

        if values is not None:
            name, xmin, ymin, xmax, ymax = values
            label_index = names.index(name)
            width, height = extract_image_dimensions(xml_content)

            normalized_values = [
                label_index,
                xmin / width,
                ymin / height,
                xmax / width,
                ymax / height
            ]
            labels.append(' '.join(str(value) for value in normalized_values))

    txt_content = '\n'.join(labels)

    os.makedirs(output_path, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(xml_file_path))[0]
    txt_file_path = os.path.join(output_path, file_name + '.txt')

    with open(txt_file_path, 'w') as txt_file:
        txt_file.write(txt_content)

    print(f"Conversion completed and saved to '{txt_file_path}'.")


def extract_object_values(object_xml):
    name_start = object_xml.find('<name>') + len('<name>')
    name_end = object_xml.find('</name>')
    name = object_xml[name_start:name_end].strip()

    xmin_start = object_xml.find('<xmin>') + len('<xmin>')
    xmin_end = object_xml.find('</xmin>')
    xmin = int(object_xml[xmin_start:xmin_end])

    ymin_start = object_xml.find('<ymin>') + len('<ymin>')
    ymin_end = object_xml.find('</ymin>')
    ymin = int(object_xml[ymin_start:ymin_end])

    xmax_start = object_xml.find('<xmax>') + len('<xmax>')
    xmax_end = object_xml.find('</xmax>')
    xmax = int(object_xml[xmax_start:xmax_end])

    ymax_start = object_xml.find('<ymax>') + len('<ymax>')
    ymax_end = object_xml.find('</ymax>')
    ymax = int(object_xml[ymax_start:ymax_end])

    return name, xmin, ymin, xmax, ymax


def extract_image_dimensions(xml_content):
    width_start = xml_content.find('<width>') + len('<width>')
    width_end = xml_content.find('</width>')
    width = int(xml_content[width_start:width_end])

    height_start = xml_content.find('<height>') + len('<height>')
    height_end = xml_content.find('</height>')
    height = int(xml_content[height_start:height_end])

    return width, height


# Example usage
xml_folder_path = './store/labels/xml/'
output_folder_path = './store/labels/txt/'
names = ['mouse']

for filename in os.listdir(xml_folder_path):
    if filename.endswith('.xml'):
        xml_file_path = os.path.join(xml_folder_path, filename)
        convert_xml_file_to_txt(xml_file_path, output_folder_path, names)
