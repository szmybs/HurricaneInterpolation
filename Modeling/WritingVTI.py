import numpy as np
import os

from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
from xml.etree.ElementTree import ElementTree

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())


def prettyXml(element, indent, newline, level = 0): 
    # 判断element是否有子元素
    if element:
        # 如果element的text没有内容      
        if element.text == None or element.text.isspace():     
            element.text = newline + indent * (level + 1)      
        else:    
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)    
    else:     
        element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level    
    temp = list(element) # 将elemnt转成list    
    for subelement in temp:    
        # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
        if temp.index(subelement) < (len(temp) - 1):     
            subelement.tail = newline + indent * (level + 1)    
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个    
            subelement.tail = newline + indent * level   
        # 对子元素进行递归操作 
        prettyXml(subelement, indent, newline, level = level + 1)
    


def WritingVTI(data, save_name):
    length = data.shape[0] - 1
    width = data.shape[1] - 1
    height = data.shape[2] - 1


    vtk_file = Element('VTKFile')
    vtk_file.set('type', 'ImageData')
    vtk_file.set('version', '1.0')
    vtk_file.set('byte_order', 'LittleEndian')
    vtk_file.set('header_type', 'UInt64')

    image_data = SubElement(vtk_file, 'ImageData')
    image_data.set('WholeExtent', '0 {height} 0 {width} 0 {length}'.format(length=length, width=width, height=height))
    image_data.set('Origin', '0 0 0')
    image_data.set('Spacing', '1.0 1.0 1.0')

    piece = SubElement(image_data, 'Piece')
    piece.set('Extent', '0 {height} 0 {width} 0 {length}'.format(length=length, width=width, height=height))

    point_data = SubElement(piece, 'PointData')
    point_data.set('Scalars', 'Scalars_')
    
    data_array = SubElement(point_data, 'DataArray')
    data_array.set('type', 'Float32')
    data_array.set('Name', 'Scalars_')
    data_array.set('format', 'ascii')
    data_array.set('RangeMin', '{min}'.format(min=np.amin(data)))
    data_array.set('RangeMax', '{max}'.format(max=np.amax(data)))

    data_str = ""
    for element in data.flat:
        data_str += (str(element) + ' ')
    data_array.text = data_str
    
    prettyXml(vtk_file, '\t', '\n')

    tree = ElementTree(vtk_file)
    tree.write(save_name, encoding = 'utf-8')



if __name__ == "__main__":
    data = np.ones(shape=(3, 3, 3)) * 0.67
    WritingVTI(data, os.path.join(os.getcwd(), "Modeling/VTIDemo.vti"))