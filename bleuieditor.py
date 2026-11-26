import bpy
import gpu
import blf
import json
import os
import math
import re
from functools import lru_cache
from time import perf_counter
from mathutils import Matrix, Vector
from gpu_extras.batch import batch_for_shader
from bpy.app.handlers import persistent
from bpy.types import NodeTree, Node, NodeSocket, Operator, Panel, UIList, Menu, PropertyGroup
from bpy.props import (
    IntVectorProperty, BoolVectorProperty,
    StringProperty, IntProperty, FloatProperty,
    FloatVectorProperty, EnumProperty, BoolProperty, PointerProperty, CollectionProperty
)
from bpy_extras.io_utils import ImportHelper, ExportHelper

def _unregister_previous_canvas_editor():
    # The persistent save handler identifies the previous script run, even
    # when Blender executed it in a different Text Editor namespace.
    for callback in tuple(bpy.app.handlers.save_pre):
        previous = getattr(callback, '__globals__', {})
        if (callback is previous.get('on_blend_save_pre')
                and previous.get('draw_armory_canvas_callback')
                and previous.get('TYPE_TO_NODE_BL_IDNAME')
                and previous.get('unregister')):
            previous['unregister']()
            break


_unregister_previous_canvas_editor()

_draw_handler = None
_UPDATING_PARENT_UI = False

# Supported extensions for automatic asset type detection
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.tga', '.bmp'}
FONT_EXTENSIONS = {'.ttf', '.otf', '.woff', '.woff2', '.ttc'}
ALL_ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | FONT_EXTENSIONS

def resolve_asset_type(filepath: str) -> str:
    """Infers whether the unified asset path points to an image or font."""
    if not filepath:
        return 'NONE'
    ext = os.path.splitext(bpy.path.abspath(filepath))[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return 'IMAGE'
    if ext in FONT_EXTENSIONS:
        return 'FONT'
    return 'NONE'

DEFAULT_LIGHT_THEME = [{
    "NAME": "Default Light",
    "WINDOW_BG_COL": -1052689,
    "WINDOW_TINT_COL": -14540254,
    "ACCENT_COL": -1118482,
    "ACCENT_HOVER_COL": -4473925,
    "ACCENT_SELECT_COL": -5592406,
    "BUTTON_COL": -3355444,
    "BUTTON_TEXT_COL": -14540254,
    "BUTTON_HOVER_COL": -5000269,
    "BUTTON_PRESSED_COL": -5131855,
    "TEXT_COL": -6710887,
    "LABEL_COL": -5592406,
    "SEPARATOR_COL": -6710887,
    "HIGHLIGHT_COL": -14656100,
    "CONTEXT_COL": -5592406,
    "PANEL_BG_COL": -5592406,
    "FONT_SIZE": 26,
    "ELEMENT_W": 200,
    "ELEMENT_H": 48,
    "ELEMENT_OFFSET": 8,
    "ARROW_SIZE": 10,
    "BUTTON_H": 44,
    "CHECK_SIZE": 30,
    "CHECK_SELECT_SIZE": 16,
    "SCROLL_W": 12,
    "TEXT_OFFSET": 16,
    "TAB_W": 24,
    "FILL_WINDOW_BG": False,
    "FILL_BUTTON_BG": True,
    "FILL_ACCENT_BG": False,
    "LINK_STYLE": 0,
    "FULL_TABS": False
}]

# Zui theme editor. Theme data remains in Armory's _themes.json schema.
_THEME_EDITING = False
THEME_COLOR_KEYS = ('TEXT_COL', 'BUTTON_COL', 'BUTTON_TEXT_COL', 'BUTTON_HOVER_COL',
                    'ACCENT_COL', 'ACCENT_HOVER_COL', 'ACCENT_SELECT_COL', 'PANEL_BG_COL',
                    'WINDOW_BG_COL', 'LABEL_COL', 'SEPARATOR_COL', 'HIGHLIGHT_COL')

def canvas_theme_list(tree):
    try:
        data = json.loads(tree.theme_json)
        if isinstance(data, list):
            themes = [x for x in data if isinstance(x, dict)]
            if themes:
                return themes
    except (TypeError, ValueError):
        pass
    return [dict(DEFAULT_LIGHT_THEME[0])]

def canvas_current_theme(tree):
    cache = _canvas_draw_cache
    if cache is not None and cache['tree'] == tree and 'theme' in cache:
        return cache['theme']
    themes = canvas_theme_list(tree)
    return next((t for t in themes if t.get('NAME') == tree.theme_name), themes[0])

def canvas_theme_text_offset(tree):
    return max(0, min(256, int(canvas_current_theme(tree).get('TEXT_OFFSET', 16))))

def canvas_text_padding(node):
    return node.text_offset if node.use_local_padding else canvas_theme_text_offset(node.id_data)


def normalize_text_lines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def get_multiline_text(node):
    return normalize_text_lines(node.text_content).replace("\\", "\\\\").replace("\n", r"\n")


def set_multiline_text(node, value):
    node.text_content = re.sub(r"\\([n\\])", lambda match: "\n" if match[1] == "n" else "\\", value)


def canvas_node_font_size(node, tree):
    if node.font_mode == 'FIXED':
        return node.font_size
    if node.font_mode == 'HEIGHT':
        return max(4, min(256, math.floor(node.height * 0.72 + 0.5)))
    return canvas_theme_font_size(tree)

def get_font_mode(node):
    stored = node.get("font_mode")
    if stored in (0, 1, 2):
        return stored
    # Match the earlier preview for existing nodes and new Text elements.
    if node.armory_type == '0':
        return 2
    return 1 if node.preview_font_override else 0

def set_font_mode(node, value):
    node["font_mode"] = value
    mark_tree_dirty(node, None)

def canvas_theme_font_size(tree):
    return max(4, min(256, int(canvas_current_theme(tree).get('FONT_SIZE', 26))))

def canvas_theme_color(tree, key):
    value = canvas_current_theme(tree).get(key)
    if not isinstance(value, int):
        value = DEFAULT_LIGHT_THEME[0][key]
    return argb_int_to_rgba(value)


def draw_canvas_pair(layout, node, prop, label, axes, *, first_index=0):
    if label:
        layout.label(text=label)
    row = layout.row(align=True)
    for index, axis in enumerate(axes, first_index):
        row.prop(node, prop, index=index, text=axis)


def canvas_element_color(node, element_type):
    key = 'TEXT_COL' if element_type == 0 else 'BUTTON_COL' if element_type in {2, 18, 19} else 'ACCENT_COL'
    color = tuple(node.element_color) if node.use_element_color else canvas_theme_color(node.id_data, key)
    session = _canvas_simulations.get(node.id_data.as_pointer())
    if element_type == 2 and node.control_variant != 'STEPPER' and session and not node.disabled:
        if session['pressed'] == node.name:
            color = node.color_press if node.use_custom_button_states else canvas_theme_color(node.id_data, 'BUTTON_PRESSED_COL')
        elif session['hover'] == node.name:
            color = node.color_hover if node.use_custom_button_states else canvas_theme_color(node.id_data, 'BUTTON_HOVER_COL')
    return canvas_grey(color) if element_type in CANVAS_DISABLED_TYPES and node.disabled else color


def canvas_control_text_color(node, element_type):
    if element_type == 0:
        return canvas_element_color(node, element_type)
    if element_type in TEXT_OVERRIDE_TYPES and node.use_custom_button_states:
        color = tuple(node.color_text)
    else:
        color = canvas_theme_color(node.id_data, 'BUTTON_TEXT_COL' if element_type == 2 else 'TEXT_COL')
    return canvas_grey(color) if element_type in CANVAS_DISABLED_TYPES and node.disabled else color


def canvas_preview_label(node):
    kind = int(node.armory_type)
    if kind not in TEXT_CAPABLE_TYPES:
        return ''
    text = node.text_content
    label = text or node.element_name
    if kind in {10, 11, 20}:
        return simulation_value(node, "input", node.preview_input_text or label)
    return label


def theme_load_fields(tree):
    global _THEME_EDITING
    theme = canvas_current_theme(tree)
    _THEME_EDITING = True
    try:
        tree.theme_edit_name = str(theme.get('NAME', 'Default Light'))
        tree.theme_font_size = canvas_theme_font_size(tree)
        tree.theme_text_offset = canvas_theme_text_offset(tree)
        for key in THEME_COLOR_KEYS:
            setattr(tree, 'theme_' + key.lower(), canvas_theme_color(tree, key))
    finally:
        _THEME_EDITING = False

def theme_field_update(self, context, changed_key=None):
    if _THEME_EDITING:
        return
    themes = canvas_theme_list(self)
    current = canvas_current_theme(self)
    for theme in themes:
        if theme.get('NAME') == current.get('NAME'):
            for key in ((changed_key,) if changed_key else THEME_COLOR_KEYS):
                theme[key] = rgba_to_argb_int(getattr(self, 'theme_' + key.lower()))
            break
    self.theme_json = json.dumps(themes, ensure_ascii=False)
    self.is_dirty = True

def theme_font_size_update(self, context):
    if _THEME_EDITING:
        return
    themes = canvas_theme_list(self)
    for theme in themes:
        if theme.get('NAME') == self.theme_name:
            theme['FONT_SIZE'] = int(self.theme_font_size)
            break
    self.theme_json = json.dumps(themes, ensure_ascii=False)
    self.is_dirty = True

def theme_text_offset_update(self, context):
    if _THEME_EDITING:
        return
    themes = canvas_theme_list(self)
    for theme in themes:
        if theme.get('NAME') == self.theme_name:
            theme['TEXT_OFFSET'] = int(self.theme_text_offset)
            break
    self.theme_json = json.dumps(themes, ensure_ascii=False)
    self.is_dirty = True

def make_theme_color_update(key):
    def update(self, context):
        theme_field_update(self, context, key)
    return update

def canvas_theme_path(tree):
    if tree.export_filepath:
        return os.path.join(os.path.dirname(os.path.abspath(bpy.path.abspath(tree.export_filepath))), '_themes.json')
    return os.path.join(get_default_armory_asset_dir(), '_themes.json')

def theme_import_file(tree, path):
    if not os.path.isfile(path):
        return False
    with open(path, 'r', encoding='utf-8-sig') as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise ValueError('Expected an Armory _themes.json array')
    tree.theme_json = json.dumps(data, ensure_ascii=False)
    if not any(t.get('NAME') == tree.theme_name for t in data) and data:
        tree.theme_name = str(data[0].get('NAME', 'Default Light'))
    theme_load_fields(tree)
    return True

TYPE_TO_NODE_BL_IDNAME = {
    0:  'ArmoryCanvasTextNode',
    1:  'ArmoryCanvasImageNode',
    2:  'ArmoryCanvasButtonNode',
    3:  'ArmoryCanvasEmptyNode',
    10: 'ArmoryCanvasTextInputNode',
    20: 'ArmoryCanvasTextAreaNode',
    6:  'ArmoryCanvasCheckNode',
    7:  'ArmoryCanvasRadioNode',
    11: 'ArmoryCanvasKeyInputNode',
    8:  'ArmoryCanvasComboBoxNode',
    9:  'ArmoryCanvasSliderNode',
    12: 'ArmoryCanvasFilledRectNode',
    13: 'ArmoryCanvasRectNode',
    14: 'ArmoryCanvasFilledCircleNode',
    15: 'ArmoryCanvasCircleNode',
    16: 'ArmoryCanvasFilledTriangleNode',
    17: 'ArmoryCanvasTriangleNode',
    18: 'ArmoryCanvasProgressNode',
    19: 'ArmoryCanvasCProgressNode',
}

NODE_GROUP_DEFAULTS = {
    'BASIC':    (0.24, 0.34, 0.46),
    'BUTTONS':  (0.18, 0.42, 0.44),
    'INPUTS':   (0.46, 0.35, 0.18),
    'SHAPES':   (0.34, 0.25, 0.48),
    'PROGRESS': (0.18, 0.44, 0.28),
}

def linear_to_srgb_channel(c):
    c = max(0.0, min(1.0, float(c)))
    if c <= 0.0031308:
        return c * 12.92
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055

def srgb_to_linear_channel(c):
    c = max(0.0, min(1.0, float(c)))
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4

def rgba_to_argb_int(rgba):
    r = int(round(linear_to_srgb_channel(rgba[0]) * 255))
    g = int(round(linear_to_srgb_channel(rgba[1]) * 255))
    b = int(round(linear_to_srgb_channel(rgba[2]) * 255))
    a = int(round(max(0.0, min(1.0, float(rgba[3]))) * 255))
    val = (a << 24) | (r << 16) | (g << 8) | b
    return val - 0x100000000 if val >= 0x80000000 else val

def argb_int_to_rgba(argb_val):
    uval = int(argb_val) & 0xFFFFFFFF
    a = ((uval >> 24) & 0xFF) / 255.0
    r = srgb_to_linear_channel(((uval >> 16) & 0xFF) / 255.0)
    g = srgb_to_linear_channel(((uval >> 8) & 0xFF) / 255.0)
    b = srgb_to_linear_channel((uval & 0xFF) / 255.0)
    return (r, g, b, a)

def display_rgba(linear_rgba):
    """Convert linear Blender color properties to display RGB, preserving alpha."""
    return (linear_to_srgb_channel(linear_rgba[0]),
            linear_to_srgb_channel(linear_rgba[1]),
            linear_to_srgb_channel(linear_rgba[2]),
            float(linear_rgba[3]))

def clean_number(val):
    r = round(float(val), 2)
    return int(r) if r.is_integer() else r

def snap_val(val, step, enabled):
    if not enabled or step <= 1:
        return int(round(val))
    return int(round(val / float(step)) * step)

def get_ordered_canvas_nodes(ntree):
    nodes = [n for n in ntree.nodes if isinstance(n, ArmoryCanvasNodeBase)]
    nodes.sort(key=lambda n: n.z_order)
    return nodes

def normalize_z_orders(ntree):
    nodes = get_ordered_canvas_nodes(ntree)
    for idx, node in enumerate(nodes):
        if node.z_order != idx:
            node.z_order = idx
    return nodes

def get_parent_node(node):
    cache = _canvas_draw_cache
    if cache is not None and cache['tree'] == node.id_data:
        return cache['parents'].get(node)
    if "Parent" in node.outputs and node.outputs["Parent"].is_linked:
        cand = node.outputs["Parent"].links[0].to_node
        if isinstance(cand, ArmoryCanvasNodeBase):
            return cand
    if "Parent" in node.inputs and node.inputs["Parent"].is_linked:
        cand = node.inputs["Parent"].links[0].from_node
        if isinstance(cand, ArmoryCanvasNodeBase):
            return cand
    return None

def get_direct_children(ntree, parent_node):
    cache = _canvas_draw_cache
    if cache is not None and cache['tree'] == ntree:
        return list(cache['children'].get(parent_node, ()))
    kids = [
        n for n in get_ordered_canvas_nodes(ntree)
        if get_parent_node(n) == parent_node
    ]
    kids.sort(key=lambda n: (n.z_order, n.name))
    return kids

LAYOUT_CONTAINER_TYPES = {3, 12, 13}

CANVAS_DISABLED_TYPES = {2, 6, 7, 9, 10}
CANVAS_STYLE_TYPES = {2, 12, 13, 14, 15, 16, 17}
CANVAS_RADIUS_TYPES = {2, 12, 13}

# Optional metadata is stored in one block so removing an override removes it
# from both new exports and round-tripped JSON.
CANVAS_FEATURES = {
    'pivot': 'pivot', 'min_size': 'min_size', 'max_size': 'max_size',
    'disabled': 'disabled', 'wrap': 'wrap_text', 'wrap_width': 'wrap_width',
    'slice_saved': 'slice_borders', 'slice_enabled': 'slice_enabled', 'slice': 'slice_borders', 'radius': 'corner_radius', 'shadow': 'shadow_enabled',
    'shadow_offset': 'shadow_offset', 'shadow_color': 'shadow_color',
    'offset_position': 'offset_position', 'offset_rotation': 'offset_rotation',
    'offset_scale': 'offset_scale_xy', 'offset_visual_only': 'offset_visual_only',
    'line': 'line_enabled', 'line_color': 'line_color', 'line_thickness': 'line_thickness',
    'line_position': 'line_position', 'text_padding_vertical': 'text_padding_vertical',
    'scroll_bar_size': 'scroll_bar_size', 'scroll_bar_color': 'scroll_bar_color',
    'scroll_track_color': 'scroll_track_color',
    'mask_children': 'mask_children', 'scroll_content': 'scroll_content', 'scroll_axes': 'scroll_axes', 'variant': 'control_variant',
    'step_min': 'step_min', 'step_max': 'step_max', 'step': 'step_amount',
    'step_value': 'step_initial', 'switch_value': 'switch_initial', 'radio_style': 'radio_style',
}


# The active draw/hit-test scope points to cached canvas-space data.
_canvas_draw_cache = None
_canvas_geometry = {}
_canvas_hierarchies = {}
_canvas_frame_samples = {}
_canvas_grid_batches = {}


def active_layout_parent(node):
    parent = get_parent_node(node)
    if (parent is not None and int(parent.armory_type) in LAYOUT_CONTAINER_TYPES
            and parent.radio_layout_mode != 'NONE' and not node.ignore_parent_layout):
        return parent
    return None


def layout_child_box(tree, node, *, preview=True, visited=None):
    parent = active_layout_parent(node)
    if parent is None or (visited is not None and parent in visited) or not node.visible:
        return None
    cache = _canvas_draw_cache
    if cache is not None and cache['tree'] == tree and preview:
        groups = cache['groups']
        if parent not in groups:
            groups[parent] = calculate_layout_group_boxes(tree, parent, preview, {node} if visited is None else visited | {node})
        return groups[parent].get(node)
    return calculate_layout_group_boxes(tree, parent, preview, {node} if visited is None else visited | {node}).get(node)


def canvas_size_limits(node, axis):
    minimum = max(1, node.min_size[axis])
    maximum = max(minimum, node.max_size[axis]) if node.max_size[axis] else 2147483647
    return minimum, maximum


def clamp_canvas_size(node, axis, size):
    minimum, maximum = canvas_size_limits(node, axis)
    return max(minimum, min(maximum, size))


def canvas_grid_boxes(parent, children, available, origin):
    columns = parent.grid_columns
    rows = max(parent.grid_rows, (len(children) + columns - 1) // columns)
    boxes = {}
    for index, child in enumerate(children):
        cell = (index % columns, index // columns)
        position, size = [], []
        for axis, count in enumerate((columns, rows)):
            gap = min(parent.radio_spacing, max(0, (available[axis] - count) // max(1, count - 1))) if count > 1 else 0
            usable = max(count, available[axis] - gap * (count - 1))
            quotient, remainder = divmod(usable, count)
            start = cell[axis] * (quotient + gap) + min(cell[axis], remainder)
            cell_size = quotient + (cell[axis] < remainder)
            child_size = clamp_canvas_size(child, axis, cell_size)
            align = int(parent.layout_child_alignment)
            factor = ((align % 3) if axis == 0 else (align // 3)) / 2
            position.append(origin[axis] + start + round((cell_size - child_size) * factor))
            size.append(child_size)
        boxes[child] = (*position, *size)
    return boxes


def grow_canvas_slots(slots, flexible, children, main, free):
    while flexible and free > 0:
        share, remainder = divmod(free, len(flexible))
        remaining = []
        distributed = 0
        for order, index in enumerate(flexible):
            room = max(0, canvas_size_limits(children[index], main)[1] - slots[index])
            extra = min(room, share + (order < remainder))
            slots[index] += extra
            distributed += extra
            if room > extra:
                remaining.append(index)
        if distributed == 0:
            break
        free -= distributed
        flexible = remaining


def calculate_layout_group_boxes(tree, parent, preview, visited):
    cache = _canvas_draw_cache
    if cache is not None and cache['tree'] == tree:
        children = cache['children'].get(parent, ())
    else:
        children = get_direct_children(tree, parent)
    children = [child for child in children if child.visible and not child.ignore_parent_layout]
    if not children:
        return {}
    if parent.layout_reverse:
        children.reverse()
    left, right, top, bottom = parent.layout_padding
    ancestors = set() if visited is None else visited
    width = canvas_element_size(parent, 'width', ancestors, preview=preview)
    height = canvas_element_size(parent, 'height', ancestors, preview=preview)
    width, height = canvas_scroll_viewport_size(parent, width, height)
    available = (max(0, width - left - right), max(0, height - top - bottom))
    available = tuple(max(size, parent.scroll_content[i] - (left + right if i == 0 else top + bottom))
                      if parent.scroll_axes[i] else size for i, size in enumerate(available))
    if parent.radio_layout_mode == 'GRID':
        return canvas_grid_boxes(parent, children, available, (left, top))
    sizes = [[clamp_canvas_size(child, 0, child.get('width', 150)),
              clamp_canvas_size(child, 1, child.get('height', 48))] for child in children]
    main = 0 if parent.radio_layout_mode == 'HORIZONTAL' else 1
    cross = 1 - main
    spacing = parent.radio_spacing
    if parent.layout_use_child_scale:
        footprints = [[extent * (1 if child.offset_visual_only else child.offset_scale_xy[axis])
                       for axis, extent in enumerate(size)]
                      for child, size in zip(children, sizes)]
        alignment = int(parent.layout_child_alignment)
        align = (alignment % 3 / 2, alignment // 3 / 2)
        occupied = sum(size[main] for size in footprints) + spacing * (len(children) - 1)
        cursor = (available[main] - occupied) * align[main]
        boxes = {}
        for child, size, footprint in zip(children, sizes, footprints):
            position = [left, top]
            position[main] += cursor + (footprint[main] - size[main]) / 2
            position[cross] += ((available[cross] - footprint[cross]) * align[cross]
                                + (footprint[cross] - size[cross]) / 2)
            boxes[child] = (*map(round, position), *size)
            cursor += footprint[main] + spacing
        return boxes
    control = parent.layout_control_size
    if parent.layout_proportional:
        count = len(children)
        minimum_slot = max(canvas_size_limits(child, main)[0] if control[main] else size[main]
                           for child, size in zip(children, sizes))
        usable = max(available[main] - spacing * (count - 1), minimum_slot * count)
        share, remainder = divmod(usable, count)
        alignment = int(parent.layout_child_alignment)
        align = (alignment % 3 / 2, alignment // 3 / 2)
        cursor = 0
        boxes = {}
        for index, (child, size) in enumerate(zip(children, sizes)):
            slots = list(available)
            slots[main] = share + (index < remainder)
            position = [left, top]
            position[main] += cursor
            for axis, slot in enumerate(slots):
                if control[axis]:
                    size[axis] = clamp_canvas_size(child, axis, slot)
                position[axis] += round((slot - size[axis]) * align[axis])
            boxes[child] = (*position, *size)
            cursor += slots[main] + spacing
        return boxes

    controlled_axes = int(control[0]) | (int(control[1]) << 1)
    stretch = [child.get('_stretch_mode', 0) & controlled_axes for child in children]
    force = parent.layout_force_expand
    alignment = int(parent.layout_child_alignment)
    align = ((alignment % 3) / 2, (alignment // 3) / 2)

    # Flexible children share space after preferred sizes and spacing are reserved.
    minimums = [canvas_size_limits(child, main)[0] for child in children]
    slots = [minimum if mask & (1 << main) else size[main]
             for size, mask, minimum in zip(sizes, stretch, minimums)]
    flexible = [i for i, mask in enumerate(stretch) if force[main] or mask & (1 << main)]
    free = available[main] - sum(slots) - spacing * (len(slots) - 1)
    if free < 0 and control[main]:
        shrink_capacity = sum(size - minimum for size, minimum in zip(slots, minimums))
        shrink = min(-free, shrink_capacity)
        # Round cumulative shares to keep the allocation exact.
        used = 0
        capacity_seen = 0
        for i, size in enumerate(slots):
            capacity_seen += size - minimums[i]
            target = round(shrink * capacity_seen / shrink_capacity) if shrink_capacity else 0
            slots[i] -= target - used
            used = target
        free += shrink
    grow_canvas_slots(slots, flexible, children, main, free)

    occupied = sum(slots) + spacing * (len(slots) - 1)
    cursor = round((available[main] - occupied) * align[main])
    cross_slot = max(1, available[cross])
    boxes = {}
    for index, child in enumerate(children):
        width_height = sizes[index]
        if stretch[index] & (1 << main) or control[main]:
            width_height[main] = slots[index]
        if stretch[index] & (1 << cross) or (control[cross] and force[cross]):
            width_height[cross] = cross_slot
        elif control[cross]:
            width_height[cross] = min(width_height[cross], cross_slot)

        width_height[cross] = clamp_canvas_size(child, cross, width_height[cross])
        coordinates = [left, top]
        coordinates[main] += cursor + round((slots[index] - width_height[main]) * align[main])
        coordinates[cross] += round((available[cross] - width_height[cross]) * align[cross])
        boxes[child] = (*coordinates, *width_height)
        cursor += slots[index] + spacing
    return boxes


def layout_child_position(tree, node, *, preview=True):
    box = layout_child_box(tree, node, preview=preview)
    return box[:2] if box is not None else None


def select_radio_in_group(ntree, container_node, selected_radio):
    for child in get_direct_children(ntree, container_node):
        if int(child.armory_type) == 7:
            child.preview_radio_selected = (child == selected_radio)

def node_has_children(node):
    if "Children" in node.inputs and node.inputs["Children"].is_linked:
        return True
    if "Children" in node.outputs and node.outputs["Children"].is_linked:
        return True
    return False

def locked_group_for(node, parent_of=get_parent_node):
    """Return the outermost locked parent that owns this element's selection."""
    group = None
    visited = {node}
    parent = parent_of(node)
    while parent is not None and parent not in visited:
        visited.add(parent)
        if parent.children_locked:
            group = parent
        parent = parent_of(parent)
    return group


def element_selection_locked(node):
    return locked_group_for(node) is not None or node.group_locked


def sync_locked_group_selection(tree):
    active = tree.nodes.active
    for node in tree.nodes:
        if isinstance(node, ArmoryCanvasNodeBase) and node.select and element_selection_locked(node):
            node.select = False
    if isinstance(active, ArmoryCanvasNodeBase):
        if element_selection_locked(active):
            tree.nodes.active = None


def set_canvas_group_locked(node, locked, include_parent=False):
    if locked_group_for(node) is not None:
        return False
    if not node_has_children(node) or include_parent:
        node.group_locked = locked
    node.children_locked = locked if node_has_children(node) else False
    for child in get_ordered_canvas_nodes(node.id_data):
        if is_ancestor_of(node, child):
            child.group_locked = locked
            child.children_locked = locked and node_has_children(child)
    if locked:
        sync_locked_group_selection(node.id_data)
    return True


def is_hidden_by_collapsed_ancestor(node):
    visited = {node}
    curr = get_parent_node(node)
    while curr is not None and curr not in visited:
        if not getattr(curr, "outliner_expanded", True):
            return True
        visited.add(curr)
        curr = get_parent_node(curr)
    return False

def get_hierarchy_depth(node):
    depth = 0
    visited = {node}
    curr = get_parent_node(node)
    while curr is not None and curr not in visited:
        visited.add(curr)
        depth += 1
        curr = get_parent_node(curr)
    return depth

def is_ancestor_of(potential_ancestor, node, parent_of=get_parent_node):
    visited = {node}
    curr = parent_of(node)
    while curr is not None and curr not in visited:
        if curr == potential_ancestor:
            return True
        visited.add(curr)
        curr = parent_of(curr)
    return False

def set_node_parent(ntree, child_node, parent_node):
    if not ntree or not child_node or "Parent" not in child_node.outputs:
        return False

    if parent_node is not None:
        if parent_node == child_node:
            return False
        if getattr(parent_node, "id_data", None) != ntree:
            return False
        if "Children" not in parent_node.inputs:
            return False
        if is_ancestor_of(child_node, parent_node):
            return False

    stretch = child_node.stretch_mode != 'NONE'
    if stretch:
        width, height = child_node.width, child_node.height

    out_sock = child_node.outputs["Parent"]
    for link in list(out_sock.links):
        ntree.links.remove(link)

    if parent_node is not None:
        children_socket = parent_node.inputs["Children"]
        if children_socket.link_limit == 1:
            children_socket.link_limit = 4095
        ntree.links.new(out_sock, children_socket)
    if stretch:
        child_node.width, child_node.height = width, height
    if locked_group_for(child_node) is not None:
        sync_locked_group_selection(ntree)
    return True

def build_hierarchical_outliner_order(ntree, all_nodes=None, parents=None):
    if all_nodes is None:
        all_nodes = get_ordered_canvas_nodes(ntree)

    parent_of = get_parent_node if parents is None else parents.get
    node_set = set(all_nodes)
    children_map = {node: [] for node in all_nodes}
    roots = []

    for node in all_nodes:
        parent = parent_of(node)
        if parent in node_set and parent != node and not is_ancestor_of(node, parent, parent_of):
            children_map[parent].append(node)
        else:
            roots.append(node)

    roots.sort(key=lambda node: (node.z_order, node.name))
    for children in children_map.values():
        children.sort(key=lambda node: (node.z_order, node.name))

    result = []
    visited = set()

    def walk(node, depth, is_last):
        if node in visited:
            return
        visited.add(node)
        result.append((node, depth, is_last))
        children = children_map.get(node, [])
        for child_index, child in enumerate(children):
            walk(child, depth + 1, child_index == len(children) - 1)

    for root_index, root in enumerate(roots):
        walk(root, 0, root_index == len(roots) - 1)

    for node in all_nodes:
        if node not in visited:
            walk(node, 0, True)

    return result

def canvas_paint_order(tree):
    """Paint back to front, keeping each parent below its descendants."""
    cache = _canvas_draw_cache
    shared = cache['topology'] if cache is not None and cache['tree'] == tree else None
    if shared is not None and 'order' in shared:
        return shared['order']
    hierarchy = shared['hierarchy'] if shared is not None else build_hierarchical_outliner_order(tree)
    nodes = {node for node, _depth, _last in hierarchy}
    order = []
    visited = set()

    def visit(node):
        if node in visited:
            return
        visited.add(node)
        parent = get_parent_node(node)
        if parent in nodes:
            visit(parent)
        order.append(node)

    for node, _depth, _last in reversed(hierarchy):
        visit(node)
    if shared is not None:
        shared['order'] = order
    return order


def get_node_collection_index(ntree, target_node):
    if not ntree or target_node is None:
        return -1
    for index, node in enumerate(ntree.nodes):
        if node == target_node:
            return index
    return -1

def sync_active_element_index(ntree, active_node):
    global _UPDATING_PARENT_UI
    if _UPDATING_PARENT_UI or not ntree or not isinstance(active_node, ArmoryCanvasNodeBase):
        return
    index = get_node_collection_index(ntree, active_node)
    if index >= 0 and ntree.active_element_index != index:
        _UPDATING_PARENT_UI = True
        try:
            ntree.active_element_index = index
        finally:
            _UPDATING_PARENT_UI = False

# -------------------------------------------------------------------------
# Shared Serialization & Deserialization Helpers (Armory Asset Alignment)
# -------------------------------------------------------------------------
def get_default_armory_asset_dir():
    """Resolves the default Armory Assets directory relative to the blend file."""
    if bpy.data.is_saved:
        blend_dir = os.path.dirname(bpy.path.abspath(bpy.data.filepath))
        # Check standard Armory project layout (Assets folder or build folder)
        assets_dir = os.path.join(blend_dir, "Assets")
        if os.path.isdir(assets_dir):
            return assets_dir
        # Check compiled assets dir
        compiled_assets = os.path.join(blend_dir, "build_" + os.path.splitext(os.path.basename(bpy.data.filepath))[0], "compiled", "Assets")
        if os.path.isdir(compiled_assets):
            return compiled_assets
        return blend_dir
    return os.path.abspath(".")


def _canvas_read_sidecar(path):
    """Armory .files is a newline-separated list of file references."""
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8-sig') as stream:
        return [line.strip() for line in stream if line.strip()]


def _canvas_resolve_asset(json_path, reference, name):
    """Resolve Armory paths against canvas location; never treat // as Blender-relative."""
    import ntpath
    base = os.path.dirname(os.path.abspath(json_path))
    ref = str(reference or '').replace('\\', '/')
    name = str(name or '')
    candidates = []
    def add(path):
        if path and path not in candidates:
            candidates.append(os.path.normpath(path))
    if ref and not ref.startswith('//'):
        if os.path.isabs(ref) or ntpath.isabs(ref):
            add(ref)
        else:
            add(os.path.join(base, ref))
    if ref.startswith('//'):
        # Armory virtual paths are not Blender's // paths.
        virtual = ref[2:]
        add(os.path.join(base, virtual))
        for ancestor in [base, *list(__import__('pathlib').Path(base).parents)]:
            add(os.path.join(str(ancestor), virtual))
    for filename in dict.fromkeys((os.path.basename(ref), os.path.basename(name.replace('\\', '/')))):
        if not filename:
            continue
        add(os.path.join(base, filename))
        add(os.path.join(base, '..', filename))
        for ancestor in [base, *list(__import__('pathlib').Path(base).parents)]:
            for folder in ('Sources', 'Bundled', 'Bundled/canvas', 'Assets', 'Assets/canvas'):
                add(os.path.join(str(ancestor), folder, filename))
    return next((v for v in candidates if os.path.isfile(v)), None)


def canvas_runtime_asset_name(node):
    # Keep an imported Armory alias while the editor shows the file basename.
    if node.asset_name == node.get('_armory_imported_display_name'):
        return node.get('_armory_imported_asset_name', node.asset_name)
    return node.asset_name


def _canvas_asset_ref(node, previous=None, canvas_dir=None):
    """Use the original Armory reference unless the user selects a new file."""
    previous = previous or {}
    raw = str(node.asset_path or '')
    name = str(canvas_runtime_asset_name(node) or '')
    old_ref = str(previous.get('file', ''))
    imported_path = str(node.get('_armory_imported_asset_path', ''))
    if old_ref and name == str(previous.get('name', '')) and raw == imported_path:
        return old_ref
    if raw:
        full = bpy.path.abspath(raw)
        if os.path.isfile(full) and canvas_dir:
            return os.path.relpath(full, canvas_dir).replace('\\', '/')
        if os.path.isfile(full):
            return os.path.abspath(full).replace('\\', '/')
    return old_ref if old_ref and name == str(previous.get('name', '')) else name

def canvas_bundle_button_image(source, canvas_dir):
    # Armory expects .files entries relative to Bundled/canvas that start with ../../.
    if (os.path.basename(canvas_dir).lower() != 'canvas'
            or os.path.basename(os.path.dirname(canvas_dir)).lower() != 'bundled'
            or not os.path.isfile(source)):
        return source
    import hashlib
    from pathlib import Path
    project = Path(canvas_dir).parent.parent
    image_dir = project / 'Assets' / 'canvas-button-images'
    input_file = Path(source)
    if input_file.parent == image_dir and input_file.name.startswith('canvas_button_'):
        return source
    data = input_file.read_bytes()
    name = 'canvas_button_' + hashlib.sha256(data).hexdigest()[:24] + input_file.suffix.lower()
    image_dir.mkdir(parents=True, exist_ok=True)
    destination = image_dir / name
    if not destination.exists() or destination.read_bytes() != data:
        destination.write_bytes(data)
    return str(destination)


def _canvas_atomic_write(path, contents):
    """Replace only after a complete successful write."""
    import tempfile
    folder = os.path.dirname(path)
    fd, temp = tempfile.mkstemp(prefix='.armory_', dir=folder)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(contents)
        if os.path.exists(path):
            import stat
            os.chmod(temp, stat.S_IMODE(os.stat(path).st_mode))
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def export_canvas_features(node, kind):
    data = {}
    if tuple(node.pivot) != (.5, .5):
        data['pivot'] = list(node.pivot)
    if any(node.min_size) or any(node.max_size):
        data.update(min_size=list(node.min_size), max_size=list(node.max_size),
                    preferred_size=[node.get('width', 150), node.get('height', 48)])
    if kind in CANVAS_DISABLED_TYPES and node.disabled:
        data['disabled'] = True
    if kind == 7:
        data['radio_style'] = node.radio_style
    if kind == 0 and node.wrap_text:
        data.update(wrap=True, wrap_width=node.wrap_width, manual_multiline=node.multiline)
    if kind in CANVAS_RADIUS_TYPES and node.corner_radius:
        data['radius'] = node.corner_radius
    if kind in CANVAS_STYLE_TYPES and node.shadow_enabled:
        data.update(shadow=True, shadow_offset=list(node.shadow_offset),
                    shadow_color=rgba_to_argb_int(node.shadow_color))
    if kind in TEXT_LAYOUT_TYPES and any(node.text_padding_vertical):
        data['text_padding_vertical'] = list(node.text_padding_vertical)
    if canvas_has_offset(node) or not node.offset_visual_only:
        data.update(offset_position=list(node.offset_position), offset_rotation=node.offset_rotation,
                    offset_scale=list(node.offset_scale_xy), offset_visual_only=node.offset_visual_only)
    if node.line_enabled:
        data.update(line=True, line_color=rgba_to_argb_int(node.line_color),
                    line_thickness=node.line_thickness, line_position=node.line_position)
    if node.mask_children:
        data.update(mask_children=True, scroll_axes=[False, False])
    if kind in LAYOUT_CONTAINER_TYPES and any(node.scroll_axes):
        data['scroll_content'] = list(node.scroll_content)
        data['scroll_axes'] = list(node.scroll_axes)
        data['scroll_bar_size'] = list(node.scroll_bar_size)
        data['scroll_bar_color'] = rgba_to_argb_int(node.scroll_bar_color)
        data['scroll_track_color'] = rgba_to_argb_int(node.scroll_track_color)
    if kind == 2 and node.control_variant != 'STEPPER' and resolve_asset_type(node.button_image_path) == 'IMAGE':
        data['button_image'] = os.path.basename(node.button_image_path.replace(chr(92), '/'))
    if kind in {1, 2} and (node.slice_enabled or any(node.slice_borders)):
        data['slice_enabled'] = node.slice_enabled
        data['slice_saved'] = list(node.slice_borders)
    if node.slice_enabled and (kind == 1 or 'button_image' in data):
        data['slice'] = list(node.slice_borders)
    if kind == 2 and node.control_variant == 'STEPPER':
        data.update(variant='STEPPER', step_min=node.step_min, step_max=max(node.step_min, node.step_max),
                    step=node.step_amount, step_value=max(node.step_min, min(max(node.step_min, node.step_max), node.step_initial)))
    elif kind == 6 and node.control_variant == 'SWITCH':
        data.update(variant='SWITCH', switch_value=node.switch_initial)
    if data.get('variant') == 'STEPPER' and node.use_end_color:
        data['end_color'] = rgba_to_argb_int(node.end_color)
    return data


def import_canvas_features(node, data):
    if not isinstance(data, dict):
        return
    if data.get('variant') == 'STEPPER' and type(data.get('end_color')) is int:
        node.end_color = argb_int_to_rgba(data['end_color'])
        node.use_end_color = True
    if data.get('wrap') is True:
        node.multiline = data.get('manual_multiline') is True
    for key, prop in CANVAS_FEATURES.items():
        if key not in data:
            continue
        value = data[key]
        if key == 'offset_scale' and type(value) in {int, float}:
            value = (value, value)
        if key in {'shadow_color', 'line_color', 'scroll_bar_color', 'scroll_track_color'}:
            if type(value) is not int:
                continue
            value = argb_int_to_rgba(value)
        try:
            setattr(node, prop, value)
        except (TypeError, ValueError, OverflowError):
            continue


def canvas_step_value(node):
    value = simulation_value(node, 'step', node.get('_preview_step_value', node.step_initial))
    return max(node.step_min, min(max(node.step_min, node.step_max), value))


def export_canvas_tree_to_json(tree, filepath):
    if not tree or tree.bl_idname != 'ArmoryCanvasTreeType':
        return None
    out_path = bpy.path.abspath(filepath) if filepath else ''
    if not out_path or os.path.isdir(out_path):
        directory = out_path if out_path and os.path.isdir(out_path) else get_default_armory_asset_dir()
        out_path = os.path.join(directory, (tree.canvas_name or tree.name) + '.json')
    elif not out_path.lower().endswith('.json'):
        out_path += '.json'
    out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    ordered = [item[0] for item in build_hierarchical_outliner_order(tree)]
    # IDs are persistent node custom properties, independent of Outliner position.
    used = set()
    for node in ordered:
        ident = node.get('_armory_canvas_id')
        if type(ident) is not int or ident < 0 or ident in used:
            ident = None
        if ident is not None:
            used.add(ident)
        else:
            node['_armory_canvas_id'] = -1
    next_id = max(used, default=-1) + 1
    for node in ordered:
        if node['_armory_canvas_id'] == -1:
            while next_id in used:
                next_id += 1
            node['_armory_canvas_id'] = next_id
            used.add(next_id)
            next_id += 1
    node_ids = {node: int(node['_armory_canvas_id']) for node in ordered}

    # Keep imported JSON fields that the editor has no UI for.
    try:
        canvas = json.loads(tree.get('_armory_canvas_original', '{}'))
    except (TypeError, ValueError):
        canvas = {}
    if not isinstance(canvas, dict):
        canvas = {}
    original_assets = canvas.get('assets', [])
    original_assets = original_assets if isinstance(original_assets, list) else []
    by_name = {a.get('name'): a for a in original_assets if isinstance(a, dict) and a.get('name')}
    assets = [dict(a) for a in original_assets if isinstance(a, dict)]
    asset_by_name = {a.get('name'): a for a in assets if a.get('name')}
    used_asset_ids = {a.get('id') for a in assets if type(a.get('id')) is int}
    next_asset_id = max(used_asset_ids, default=-1) + 1

    def register_asset(name, file):
        nonlocal next_asset_id
        asset = asset_by_name.get(name)
        if asset is None:
            asset = {'name': name, 'id': next_asset_id}
            next_asset_id += 1
            assets.append(asset)
            asset_by_name[name] = asset
        asset['file'] = file

    elements = []
    for node in canvas_paint_order(tree):
        ident = node_ids[node]
        parent = get_parent_node(node)
        parent_id = node_ids.get(parent)
        current = node.to_armory_dict(ident, parent_id)
        try:
            original = json.loads(node.get('_armory_canvas_original_element', '{}'))
        except (TypeError, ValueError):
            original = {}
        if not isinstance(original, dict):
            original = {}
        original.update(current)
        # Remove imported overrides if the corresponding node override is disabled.
        if not node.use_element_color:
            original.pop('color', None)
            if int(node.armory_type) == 0:
                original.pop('color_text', None)
        if not node.use_custom_button_states:
            for key in ('color_hover', 'color_press'):
                original.pop(key, None)
            if int(node.armory_type) != 0:
                original.pop('color_text', None)
        if int(node.armory_type) in TEXT_OVERRIDE_TYPES - TEXT_HOVER_TYPES:
            original.pop('color_hover', None)
        if not node.use_progress_color:
            original.pop('color_progress', None)
        # Size is only present for Fixed Size. Remove stale settings on re-export.
        for key in ('font_mode', 'font_size', 'text_offset', 'multiline', 'text_vertical_align'):
            if key not in current:
                original.pop(key, None)
        original.pop('font', None)
        for key in ('_editor_layout_manual', '_editor_layout_group', '_editor_layout_preferred_size', '_editor_layout_stretch'):
            if key not in current:
                original.pop(key, None)
        if '_editor_stretch' not in current:
            original.pop('_editor_stretch', None)
        if '_editor_features' not in current:
            original.pop('_editor_features', None)
        original['id'] = ident
        original['parent'] = parent_id
        original['children'] = []
        elements.append(original)
        if current['asset']:
            name = current['asset']
            register_asset(name, _canvas_asset_ref(node, by_name.get(name), out_dir))
        features = current.get('_editor_features', {})
        name = features.get('button_image')
        if name:
            full = canvas_bundle_button_image(bpy.path.abspath(node.button_image_path), out_dir)
            name = os.path.basename(full)
            features['button_image'] = name
            try:
                file = os.path.relpath(full, out_dir).replace(chr(92), '/')
            except ValueError:
                file = full.replace(chr(92), '/')
            register_asset(name, file)
    element_by_id = {e['id']: e for e in elements}
    for elem in elements:
        parent = element_by_id.get(elem['parent'])
        if parent is not None:
            parent['children'].append(elem['id'])

    canvas.update({
        'name': tree.canvas_name, 'width': int(tree.canvas_width),
        'height': int(tree.canvas_height), 'theme': tree.theme_name,
        'visible': bool(tree.canvas_visible), 'elements': elements,
        'assets': assets,
    })
    # The Armory exporter uses the Blender scene's render percentage.
    canvas.pop('_canvas_resolution_percentage', None)
    canvas.setdefault('x', 0)
    canvas.setdefault('y', 0)
    # Sidecar must contain raw file references, one per line, NOT JSON.
    refs = [str(a.get('file', '')) for a in assets if a.get('file')]
    # Preserve sidecar-only entries from the imported project.
    try:
        extra_refs = json.loads(tree.get('_armory_canvas_sidecar', '[]'))
        if isinstance(extra_refs, list):
            refs.extend(str(v) for v in extra_refs if isinstance(v, str))
    except (ValueError, TypeError):
        pass
    refs = list(dict.fromkeys(refs))
    json_text = json.dumps(canvas, ensure_ascii=False, separators=(',', ':')) + '\n'
    _canvas_atomic_write(out_path, json_text)
    _canvas_atomic_write(os.path.splitext(out_path)[0] + '.files',
                         ''.join(ref + '\n' for ref in refs))
    theme_path = os.path.join(out_dir, '_themes.json')
    _canvas_atomic_write(theme_path, json.dumps(canvas_theme_list(tree), ensure_ascii=False, indent=2) + '\n')
    tree['_armory_canvas_original'] = json_text
    tree['_armory_canvas_sidecar'] = json.dumps(refs)
    for node in ordered:
        if node.asset_name:
            node['_armory_imported_asset_path'] = node.asset_path
    tree.is_dirty = False
    return out_path

def import_canvas_json_to_tree(tree, filepath):
    if not tree or tree.bl_idname != 'ArmoryCanvasTreeType' or not filepath:
        return 0

    abs_path = os.path.abspath(bpy.path.abspath(filepath))
    with open(abs_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(data.get('elements', []), list):
        raise ValueError('Invalid Armory Canvas JSON: expected object with elements list')
    tree['_armory_canvas_original'] = json.dumps(data, ensure_ascii=False)
    sidecar_path = os.path.splitext(abs_path)[0] + '.files'
    tree['_armory_canvas_sidecar'] = json.dumps(_canvas_read_sidecar(sidecar_path))

    tree.nodes.clear()
    tree.links.clear()

    tree.canvas_name = str(data.get("name", "UI"))
    tree.canvas_width = max(64, int(data.get("width", 1024)))
    tree.canvas_height = max(64, int(data.get("height", 1024)))
    tree.pop('_canvas_view_top', None)
    tree.theme_name = str(data.get("theme", "Default Light"))
    tree.canvas_visible = bool(data.get("visible", True))
    percentage = data.get('_canvas_resolution_percentage')
    if type(percentage) is int and 1 <= percentage <= 100:
        tree.canvas_resolution_scale = percentage
    theme_import_file(tree, os.path.join(os.path.dirname(abs_path), "_themes.json"))

    asset_file_map = {}
    for asset in data.get('assets', []):
        if isinstance(asset, dict) and asset.get('name'):
            asset_file_map[str(asset['name'])] = str(asset.get('file', ''))

    raw_elements = data.get("elements", [])
    outliner_elements = list(reversed(raw_elements))

    id_to_node = {}
    valid_aligns = {'0', '1', '2'}
    valid_anchors = {str(i) for i in range(9)}

    for z_idx, elem in enumerate(outliner_elements):
        t_int = int(elem.get("type", 12))
        bl_idname = TYPE_TO_NODE_BL_IDNAME.get(t_int, 'ArmoryCanvasFilledRectNode')
        node = tree.nodes.new(type=bl_idname)

        if type(elem.get('id')) is int and elem['id'] >= 0:
            node['_armory_canvas_id'] = elem['id']
        node['_armory_canvas_original_element'] = json.dumps(elem, ensure_ascii=False)
        node.z_order = z_idx
        if str(t_int) in dict((item[0], item[1]) for item in ELEMENT_TYPE_ITEMS):
            node.armory_type = str(t_int)

        node.element_name = str(elem.get("name", node.bl_label))
        node.event_name = str(elem.get("event", ""))
        node.pos_x = int(elem.get("x", 0))
        node.pos_y = int(elem.get("y", 0))
        node.width = max(1, int(elem.get("width", 100)))
        node.height = max(1, int(elem.get("height", 40)))
        angle = float(elem.get("rotation", 0.0))
        node.rotation = angle if elem.get("_editor_rotation_unit") == "radians" else math.radians(angle)
        node.visible = bool(elem.get("visible", True))

        node.text_content = str(elem.get("text", ""))
        if t_int in {10, 11, 20}:
            node.preview_input_text = node.text_content
        if t_int in TEXT_CAPABLE_TYPES:
            mode = elem.get("font_mode")
            if mode in ('THEME', 'FIXED', 'HEIGHT'):
                node.font_mode = mode
            size = elem.get("font_size")
            if type(size) is int:
                node.font_size = max(4, min(256, size))
        if t_int == 0:
            node.multiline = elem.get("multiline", "\n" in normalize_text_lines(node.text_content)) is True
        if t_int in TEXT_LAYOUT_TYPES:
            vertical = elem.get("text_vertical_align")
            if vertical in ('TOP', 'MIDDLE', 'BOTTOM'):
                node.text_vertical_align = vertical
            offset = elem.get("text_offset")
            if type(offset) is int:
                node.use_local_padding = True
                node.text_offset = max(0, min(256, offset))

        # Armory's element.asset is the resource name for BOTH images and fonts.
        # The asset table's file entry is relative to the canvas JSON directory.
        a_name = str(elem.get('asset') or elem.get('font') or '')
        ref = asset_file_map.get(a_name) or a_name
        resolved = _canvas_resolve_asset(abs_path, ref, a_name) if a_name else None
        if resolved:
            # Assigning the path loads the Blender asset and sets its display name.
            node.asset_path = resolved
        else:
            node.asset_path = ''
            node.asset_name = os.path.basename(ref.replace('\\', '/'))
            node.asset_type = resolve_asset_type(ref)
            if a_name:
                print('Armory Canvas: asset not found: %s (reference: %s, canvas: %s)' % (a_name, ref, abs_path))
        if a_name:
            node['_armory_imported_asset_name'] = a_name
            node['_armory_imported_display_name'] = node.asset_name
        node['_armory_imported_asset_path'] = node.asset_path

        node.progress_total = max(1, int(elem.get("progress_total", 100)))
        node.progress_at = max(0, min(node.progress_total, int(elem.get("progress_at", 0))))
        node.strength = max(0.1, float(elem.get("strength", 1.0)))

        align_str = str(elem.get("alignment", 0))
        node.alignment = align_str if align_str in valid_aligns else '0'

        anchor_str = str(elem.get("anchor", 0))
        node.anchor = anchor_str if anchor_str in valid_anchors else '0'

        col_val = elem.get("color")
        ct_val = elem.get("color_text")
        if col_val is not None:
            node.use_element_color = True
            node.element_color = argb_int_to_rgba(col_val)
        elif t_int == 0 and ct_val is not None:
            node.use_element_color = True
            node.element_color = argb_int_to_rgba(ct_val)
        else:
            node.use_element_color = False

        if t_int in {18, 19}:
            cp_val = elem.get("color_progress")
            if cp_val is not None:
                node.use_progress_color = True
                node.color_progress = argb_int_to_rgba(cp_val)
            else:
                node.use_progress_color = False

        if t_int in TEXT_OVERRIDE_TYPES:
            ct = elem.get("color_text")
            ch = elem.get("color_hover")
            cp = elem.get("color_press")
            if ct is not None or ch is not None or cp is not None:
                node.use_custom_button_states = True
                if ct is not None:
                    node.color_text = argb_int_to_rgba(ct)
                if ch is not None:
                    node.color_hover = argb_int_to_rgba(ch)
                if cp is not None:
                    node.color_press = argb_int_to_rgba(cp)
            else:
                node.use_custom_button_states = False

        features = elem.get('_editor_features')
        import_canvas_features(node, features)
        if t_int == 2 and isinstance(features, dict) and isinstance(features.get('button_image'), str):
            name = features['button_image']
            ref = asset_file_map.get(name) or name
            node.button_image_path = _canvas_resolve_asset(abs_path, ref, name) or ref

        group = elem.get('_editor_layout_group')
        if isinstance(group, dict):
            mode = group.get('mode')
            if mode in ('NONE', 'VERTICAL', 'HORIZONTAL', 'GRID'):
                node.radio_layout_mode = mode
            for key in ('grid_rows', 'grid_columns'):
                if type(group.get(key)) is int:
                    setattr(node, key, group[key])
            if type(group.get('spacing')) is int:
                node.radio_spacing = max(0, min(512, group['spacing']))
            padding = group.get('padding')
            if isinstance(padding, list) and len(padding) == 4 and all(type(v) is int for v in padding):
                node.layout_padding = [max(0, min(2147483647, v)) for v in padding]
            alignment = group.get('alignment')
            if isinstance(alignment, str) and alignment in valid_anchors:
                node.layout_child_alignment = alignment
            node.layout_proportional = group.get('proportional') is True
            node.layout_use_child_scale = group.get('use_child_scale') is True
            node.layout_reverse = group.get('reverse') is True
            node.radio_simulate = group.get('radio') is True
            node.ignore_parent_layout = group.get('ignore') is True
            for key, prop in (('control_size', 'layout_control_size'), ('force_expand', 'layout_force_expand')):
                flags = group.get(key)
                if isinstance(flags, list) and len(flags) == 2 and all(type(flag) is bool for flag in flags):
                    setattr(node, prop, flags)
        preferred = elem.get('_editor_layout_preferred_size')
        if preferred is None and isinstance(elem.get('_editor_features'), dict):
            preferred = elem['_editor_features'].get('preferred_size')
        if isinstance(preferred, list) and len(preferred) == 2 and all(type(v) is int and 1 <= v <= 2147483647 for v in preferred):
            node.width, node.height = preferred
        manual = elem.get('_editor_layout_manual')
        if isinstance(manual, list) and len(manual) == 3 and all(type(v) is int for v in manual):
            node.pos_x, node.pos_y = manual[:2]
            if str(manual[2]) in valid_anchors:
                node.anchor = str(manual[2])

        col_i = z_idx % 3
        row_i = z_idx // 3
        node.location = (float(tree.canvas_width) + 120.0 + col_i * 260.0, canvas_view_top(tree) - row_i * 420.0)

        elem_id = elem.get("id")
        if elem_id is not None:
            id_to_node[elem_id] = node

    for elem in outliner_elements:
        elem_id = elem.get("id")
        parent_id = elem.get("parent")
        if elem_id in id_to_node and parent_id is not None and parent_id in id_to_node:
            set_node_parent(tree, id_to_node[elem_id], id_to_node[parent_id])

    # Restore stretch only after all supporting parent links exist.
    for elem in outliner_elements:
        node = id_to_node.get(elem.get("id"))
        stretch = elem.get('_editor_layout_stretch', elem.get('_editor_stretch'))
        if node is None or not isinstance(stretch, dict):
            continue
        mode = STRETCH_VALUES.get(stretch.get('mode')) if isinstance(stretch.get('mode'), str) else None
        offsets = stretch.get('offsets')
        if mode and isinstance(offsets, list) and len(offsets) == 2 and all(type(v) is int for v in offsets):
            node['_stretch_mode'] = mode
            node['_stretch_width'], node['_stretch_height'] = offsets

    normalize_z_orders(tree)
    if outliner_elements:
        first_node = id_to_node.get(outliner_elements[0].get("id"))
        if first_node is None:
            first_node = get_ordered_canvas_nodes(tree)[0]
        for node in tree.nodes:
            node.select = (node == first_node)
        tree.nodes.active = first_node
        sync_active_element_index(tree, first_node)
    else:
        tree.active_element_index = 0
    tree.is_dirty = False
    return len(outliner_elements)

def auto_export_all_dirty_canvases(force=False):
    exported = []
    for ng in bpy.data.node_groups:
        if ng.bl_idname == 'ArmoryCanvasTreeType':
            if ng.auto_save_on_play and ng.export_filepath:
                if force or ng.is_dirty:
                    out = export_canvas_tree_to_json(ng, ng.export_filepath)
                    if out:
                        exported.append(out)
    return exported

@persistent
def on_blend_save_pre(filepath):
    # Linked editor texts have a disk target. Internal texts are saved by Blender.
    for script in bpy.data.texts:
        if not script.filepath or not script.is_dirty or script.is_modified:
            continue
        source = script.as_string()
        if "class ArmoryCanvasTree(NodeTree)" not in source or "def draw_armory_canvas_callback(" not in source:
            continue
        blend_path = bpy.data.filepath or filepath
        path = bpy.path.abspath(script.filepath, start=os.path.dirname(blend_path) if blend_path else None)
        try:
            _canvas_atomic_write(path, source)
        except OSError as error:
            print(f"Armory Canvas: could not save script {path}: {error}")
    auto_export_all_dirty_canvases(force=True)

@persistent
def on_depsgraph_update(scene, depsgraph):
    for update in depsgraph.updates:
        if isinstance(update.id, ArmoryCanvasTree):
            invalidate_canvas_geometry(update.id.original)
            if not update.id.is_dirty:
                update.id.is_dirty = True
            sync_locked_group_selection(update.id)
            active_node = update.id.nodes.active
            if isinstance(active_node, ArmoryCanvasNodeBase):
                sync_active_element_index(update.id, active_node)

# -------------------------------------------------------------------------
# 1. Custom Node Tree & Socket
# -------------------------------------------------------------------------
def invalidate_canvas_geometry(tree):
    key = tree.as_pointer()
    _canvas_geometry.pop(tree.session_uid, None)
    session = _canvas_simulations.get(key)
    if session is not None:
        session.pop('hit_index', None)


def invalidate_canvas_hierarchy(tree):
    _canvas_hierarchies.pop(tree.session_uid, None)
    invalidate_canvas_geometry(tree)


def update_preview_scroll(node, _context):
    invalidate_canvas_geometry(node.id_data)


def mark_canvas_order_dirty(node, context):
    invalidate_canvas_hierarchy(node.id_data)
    mark_tree_dirty(node, context)


@persistent
def reset_canvas_caches(*_args):
    _canvas_geometry.clear()
    _canvas_hierarchies.clear()
    _canvas_frame_samples.clear()
    _canvas_grid_batches.clear()
    for session in _canvas_simulations.values():
        session.pop('hit_index', None)
    canvas_text_dimensions.cache_clear()
    canvas_wrapped_text.cache_clear()


def update_canvas_debug_fps(tree, context):
    _canvas_frame_samples.clear()
    if context and context.area:
        context.area.tag_redraw()


def mark_tree_dirty(self, context):
    if isinstance(self, ArmoryCanvasTree):
        invalidate_canvas_geometry(self)
        self.is_dirty = True
    elif hasattr(self, "id_data") and isinstance(self.id_data, ArmoryCanvasTree):
        invalidate_canvas_geometry(self.id_data)
        self.id_data.is_dirty = True

def apply_group_node_colors(self, context):
    if not isinstance(self, ArmoryCanvasTree):
        return
    group_map = {
        'BASIC':    self.node_color_basic,
        'BUTTONS':  self.node_color_buttons,
        'INPUTS':   self.node_color_inputs,
        'SHAPES':   self.node_color_shapes,
        'PROGRESS': self.node_color_progress,
    }
    for node in self.nodes:
        if isinstance(node, ArmoryCanvasNodeBase):
            grp = node.node_group_category
            if grp in group_map:
                node.use_custom_color = self.color_code_nodes
                node.color = tuple(group_map[grp])
    if context and context.area:
        context.area.tag_redraw()

def on_active_index_update(self, context):
    global _UPDATING_PARENT_UI
    if _UPDATING_PARENT_UI:
        return
    nodes = self.nodes
    index = self.active_element_index
    if 0 <= index < len(nodes):
        target = nodes[index]
        if not isinstance(target, ArmoryCanvasNodeBase):
            return
        group = locked_group_for(target)
        if group is not None:
            target = group
        _UPDATING_PARENT_UI = True
        try:
            target.select = not element_selection_locked(target)
            nodes.active = target
        finally:
            _UPDATING_PARENT_UI = False
        if group is not None:
            sync_active_element_index(self, target)

        if context:
            for screen in bpy.data.screens:
                for area in screen.areas:
                    if area.type == 'NODE_EDITOR':
                        area.tag_redraw()

def get_parent_selector_candidates(ntree):
    if not ntree or getattr(ntree, "bl_idname", "") != 'ArmoryCanvasTreeType':
        return []

    active = ntree.nodes.active
    if not isinstance(active, ArmoryCanvasNodeBase):
        return []

    candidates = []
    for node in get_ordered_canvas_nodes(ntree):
        if node == active or is_ancestor_of(active, node):
            continue
        candidates.append(node)
    return candidates

def get_parent_selector_items(self, context):
    items = [('NONE', "None (Canvas Root)", "No parent element", 0, 0)]
    ntree = self if getattr(self, "bl_idname", "") == 'ArmoryCanvasTreeType' else None
    if ntree is None:
        ntree = getattr(getattr(context, "space_data", None), "node_tree", None)

    for index, node in enumerate(get_parent_selector_candidates(ntree), start=1):
        items.append((node.name, node.element_name,
                      f"Parent to {node.element_name}", 0, index))
    return items

def get_parent_selector_value(self):
    active = self.nodes.active
    if not isinstance(active, ArmoryCanvasNodeBase):
        return 0
    parent = get_parent_node(active)
    if parent is None:
        return 0
    for index, candidate in enumerate(get_parent_selector_candidates(self), start=1):
        if candidate == parent:
            return index
    return 0

def set_parent_selector_value(self, value):
    active = self.nodes.active
    if not isinstance(active, ArmoryCanvasNodeBase):
        return
    candidates = get_parent_selector_candidates(self)
    target_parent = candidates[value - 1] if 1 <= value <= len(candidates) else None
    if set_node_parent(self, active, target_parent):
        self.is_dirty = True
        sync_active_element_index(self, active)
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == 'NODE_EDITOR':
                    area.tag_redraw()

CANVAS_SIZE_PRESETS = (
    ("Portrait", 1080, 1920),
    ("Landscape", 1920, 1080),
    ("Square", 1024, 1024),
    ("Custom", 0, 0),
)


def canvas_preset_file(*, create=False):
    folder = bpy.utils.user_resource('CONFIG', path='armory_canvas', create=create)
    if not folder:
        raise OSError("Blender's user configuration folder is unavailable")
    return os.path.join(folder, 'resolution_presets.json')


CANVAS_PRESET_CATEGORIES = ('Mobile', 'Tablet', 'Desktop')


def canvas_preset_category(width, height):
    ratio = min(width, height) / max(width, height)
    if height > width and ratio < 0.65:
        return 'Mobile'
    return 'Tablet' if ratio >= 0.65 else 'Desktop'


def read_external_canvas_presets(reserved_names):
    try:
        with open(canvas_preset_file(), encoding='utf-8-sig') as stream:
            entries = json.load(stream)
    except FileNotFoundError:
        return [], False
    except (OSError, ValueError):
        return [], True

    if not isinstance(entries, list):
        return [], True
    presets = []
    invalid_entry = False
    seen = set(reserved_names)
    for entry in entries:
        if not isinstance(entry, dict):
            invalid_entry = True
            continue
        name, width, height = (entry.get(key) for key in ('name', 'width', 'height'))
        if (not isinstance(name, str) or not name.strip()
                or not all(type(size) is int and 64 <= size <= 2147483647 for size in (width, height))):
            invalid_entry = True
            continue
        name = name.strip()
        key = name.casefold()
        if key in seen:
            invalid_entry = True
            continue
        seen.add(key)
        category = entry.get('category')
        category = category.strip().title() if isinstance(category, str) else ''
        if category not in CANVAS_PRESET_CATEGORIES:
            category = canvas_preset_category(width, height)
        presets.append((name, width, height, category))
    return presets, invalid_entry


class ArmoryCanvasSizePreset(PropertyGroup):
    width: IntProperty(default=1024, min=64)
    height: IntProperty(default=1024, min=64)


def canvas_oriented_size(tree):
    width, height = tree.canvas_width, tree.canvas_height
    return (height, width) if tree.canvas_resolution_rotated else (width, height)


def canvas_preview_size(tree):
    percentage = tree.canvas_resolution_scale
    return tuple(
        max(1, (size * percentage + 50) // 100)
        for size in canvas_oriented_size(tree)
    )


def sync_armory_render_scale(tree, context):
    scene = getattr(context, 'scene', None)
    if scene is None:
        return

    owner_key = '_armory_canvas_render_scale_owner'
    previous_key = '_armory_canvas_render_scale_previous'
    applied_key = '_armory_canvas_render_scale_applied'
    owner = scene.get(owner_key)
    render = scene.render
    fields = ('resolution_x', 'resolution_y', 'resolution_percentage')
    current = tuple(getattr(render, field) for field in fields)
    previous = scene.get(previous_key, current)
    applied = scene.get(applied_key, current)
    # Upgrade the former percentage-only snapshot without losing its original value.
    if isinstance(previous, int) and isinstance(applied, int):
        previous = (*current[:2], previous)
        applied = (*current[:2], applied)

    if tree.apply_resolution_scale_to_armory:
        scene[previous_key] = [
            old if owner is not None and actual == last else actual
            for actual, old, last in zip(current, previous, applied)
        ]
        scene[owner_key] = tree
        requested = (*canvas_oriented_size(tree), tree.canvas_resolution_scale)
        for field, value in zip(fields, requested):
            setattr(render, field, value)
        scene[applied_key] = [getattr(render, field) for field in fields]
    elif owner == tree:
        for field, actual, old, last in zip(fields, current, previous, applied):
            if actual == last:
                setattr(render, field, old)
        for key in (owner_key, previous_key, applied_key):
            scene.pop(key, None)


def redraw_canvas_resolution(tree, context):
    invalidate_canvas_geometry(tree)
    if tree.apply_resolution_scale_to_armory:
        sync_armory_render_scale(tree, context)
    if context and context.area:
        context.area.tag_redraw()


def update_canvas_dimensions(tree, context):
    mark_tree_dirty(tree, context)
    redraw_canvas_resolution(tree, context)


def canvas_view_top(tree):
    return float(tree.get('_canvas_view_top', tree.canvas_height))


def set_canvas_height(tree, height):
    if '_canvas_view_top' not in tree:
        tree['_canvas_view_top'] = tree.canvas_height
    tree['canvas_height'] = height


class ArmoryCanvasTree(NodeTree):
    bl_idname = 'ArmoryCanvasTreeType'
    bl_label = 'Armory Canvas Editor'
    bl_icon = 'UV'

    def update(self):
        invalidate_canvas_hierarchy(self)

    canvas_name: StringProperty(name="Name", default="UI", update=mark_tree_dirty)
    canvas_width: IntProperty(name="Width", default=1024, min=64, update=update_canvas_dimensions)
    canvas_height: IntProperty(
        name="Height", default=1024, min=64,
        get=lambda tree: tree.get('canvas_height', 1024), set=set_canvas_height,
        update=update_canvas_dimensions,
    )
    canvas_resolution_rotated: BoolProperty(
        name="Rotate Resolution 90°",
        description="Swap the preview width and height to simulate a device turning",
        default=False, update=redraw_canvas_resolution,
    )
    canvas_resolution_scale: IntProperty(
        name="Scale",
        description="Preview the canvas at a percentage of its full resolution",
        default=100, min=1, max=100, subtype='PERCENTAGE',
        update=redraw_canvas_resolution,
    )
    apply_resolution_scale_to_armory: BoolProperty(
        name="Apply Resolution in Armory",
        description="Sync width, height, orientation and scale to the scene render settings that Armory reads",
        default=False, update=sync_armory_render_scale,
    )
    canvas_size_presets: CollectionProperty(type=ArmoryCanvasSizePreset)
    theme_name: StringProperty(name="Theme", default="Default Light", update=mark_tree_dirty)
    theme_font_size: IntProperty(name='FONT_SIZE', default=26, min=4, max=256, update=theme_font_size_update)
    theme_text_offset: IntProperty(name='TEXT_OFFSET', default=16, min=0, max=256, update=theme_text_offset_update)
    theme_json: StringProperty(name="Zui Themes JSON", default=json.dumps(DEFAULT_LIGHT_THEME))
    theme_edit_name: StringProperty(name="Theme Name", default="Default Light")
    theme_text_col: FloatVectorProperty(name="TEXT_COL", subtype='COLOR', size=4, default=argb_int_to_rgba(DEFAULT_LIGHT_THEME[0]['TEXT_COL']), min=0.0, max=1.0, update=make_theme_color_update('TEXT_COL'))
    theme_button_col: FloatVectorProperty(name="BUTTON_COL", subtype='COLOR', size=4, default=argb_int_to_rgba(DEFAULT_LIGHT_THEME[0]['BUTTON_COL']), min=0.0, max=1.0, update=make_theme_color_update('BUTTON_COL'))
    theme_button_text_col: FloatVectorProperty(name="BUTTON_TEXT_COL", subtype='COLOR', size=4, default=argb_int_to_rgba(DEFAULT_LIGHT_THEME[0]['BUTTON_TEXT_COL']), min=0.0, max=1.0, update=make_theme_color_update('BUTTON_TEXT_COL'))
    theme_button_hover_col: FloatVectorProperty(name="BUTTON_HOVER_COL", subtype='COLOR', size=4, default=argb_int_to_rgba(DEFAULT_LIGHT_THEME[0]['BUTTON_HOVER_COL']), min=0.0, max=1.0, update=make_theme_color_update('BUTTON_HOVER_COL'))
    theme_accent_col: FloatVectorProperty(name="ACCENT_COL", subtype='COLOR', size=4, default=argb_int_to_rgba(DEFAULT_LIGHT_THEME[0]['ACCENT_COL']), min=0.0, max=1.0, update=make_theme_color_update('ACCENT_COL'))
    theme_accent_hover_col: FloatVectorProperty(name="ACCENT_HOVER_COL", subtype='COLOR', size=4, default=argb_int_to_rgba(DEFAULT_LIGHT_THEME[0]['ACCENT_HOVER_COL']), min=0.0, max=1.0, update=make_theme_color_update('ACCENT_HOVER_COL'))
    theme_accent_select_col: FloatVectorProperty(name="ACCENT_SELECT_COL", subtype='COLOR', size=4, default=argb_int_to_rgba(DEFAULT_LIGHT_THEME[0]['ACCENT_SELECT_COL']), min=0.0, max=1.0, update=make_theme_color_update('ACCENT_SELECT_COL'))
    theme_panel_bg_col: FloatVectorProperty(name="PANEL_BG_COL", subtype='COLOR', size=4, default=argb_int_to_rgba(DEFAULT_LIGHT_THEME[0]['PANEL_BG_COL']), min=0.0, max=1.0, update=make_theme_color_update('PANEL_BG_COL'))
    theme_window_bg_col: FloatVectorProperty(name="WINDOW_BG_COL", subtype='COLOR', size=4, default=argb_int_to_rgba(DEFAULT_LIGHT_THEME[0]['WINDOW_BG_COL']), min=0.0, max=1.0, update=make_theme_color_update('WINDOW_BG_COL'))
    theme_label_col: FloatVectorProperty(name="LABEL_COL", subtype='COLOR', size=4, default=argb_int_to_rgba(DEFAULT_LIGHT_THEME[0]['LABEL_COL']), min=0.0, max=1.0, update=make_theme_color_update('LABEL_COL'))
    theme_separator_col: FloatVectorProperty(name="SEPARATOR_COL", subtype='COLOR', size=4, default=argb_int_to_rgba(DEFAULT_LIGHT_THEME[0]['SEPARATOR_COL']), min=0.0, max=1.0, update=make_theme_color_update('SEPARATOR_COL'))
    theme_highlight_col: FloatVectorProperty(name="HIGHLIGHT_COL", subtype='COLOR', size=4, default=argb_int_to_rgba(DEFAULT_LIGHT_THEME[0]['HIGHLIGHT_COL']), min=0.0, max=1.0, update=make_theme_color_update('HIGHLIGHT_COL'))

    canvas_visible: BoolProperty(
        name="Canvas Visible",
        description="Root visibility of the Canvas in Armory (must be enabled to render in Armory2D)",
        default=True,
        update=mark_tree_dirty
    )

    canvas_bg_color: FloatVectorProperty(
        name="Canvas Area Color",
        subtype='COLOR',
        size=4,
        default=(0.6667, 0.6667, 0.6667, 0.2),
        min=0.0, max=1.0
    )
    canvas_border_color: FloatVectorProperty(
        name="Border Line Color",
        subtype='COLOR',
        size=4,
        default=(0.8745, 0.8745, 0.8745, 1.0),
        min=0.0, max=1.0
    )

    proportional_gizmo_scale: BoolProperty(name="Proportional Scale", default=False)
    show_canvas_overlay: BoolProperty(name="Show Visual Canvas", default=True)
    show_element_names: BoolProperty(
        name="Element Name",
        description="Show the name above the active selected element",
        default=True,
    )
    smart_snap: BoolProperty(name="Smart Alignment", description="Snap to nearby sibling, parent and canvas edges or centers", default=True)
    smart_snap_distance: IntProperty(name="Guide Distance", description="Magnet radius in screen pixels", default=8, min=2, max=32)
    snap_scale: BoolProperty(name="Snap Scale", description="Snap element dimensions to the grid size", default=False)
    snap_rotation: BoolProperty(name="Snap Rotation", default=True)
    rotation_snap_degrees: FloatProperty(name="Rotation Step (°)", default=15.0, min=0.1, max=360.0)
    show_gizmos: BoolProperty(name="Show Viewport Gizmos", default=True)
    show_viewport_fps: BoolProperty(
        name="Viewport FPS",
        description="Show the Canvas viewport redraw rate during interaction",
        default=False, options={'SKIP_SAVE'}, update=update_canvas_debug_fps,
    )

    color_code_nodes: BoolProperty(
        name="Nodes Groups",
        description="Apply group header colors to nodes in the Node Editor",
        default=True,
        update=apply_group_node_colors
    )
    node_color_basic: FloatVectorProperty(
        name="Basic", subtype='COLOR', size=3,
        default=NODE_GROUP_DEFAULTS['BASIC'], min=0.0, max=1.0,
        update=apply_group_node_colors
    )
    node_color_buttons: FloatVectorProperty(
        name="Buttons", subtype='COLOR', size=3,
        default=NODE_GROUP_DEFAULTS['BUTTONS'], min=0.0, max=1.0,
        update=apply_group_node_colors
    )
    node_color_inputs: FloatVectorProperty(
        name="Inputs", subtype='COLOR', size=3,
        default=NODE_GROUP_DEFAULTS['INPUTS'], min=0.0, max=1.0,
        update=apply_group_node_colors
    )
    node_color_shapes: FloatVectorProperty(
        name="Shapes", subtype='COLOR', size=3,
        default=NODE_GROUP_DEFAULTS['SHAPES'], min=0.0, max=1.0,
        update=apply_group_node_colors
    )
    node_color_progress: FloatVectorProperty(
        name="Progress Bars", subtype='COLOR', size=3,
        default=NODE_GROUP_DEFAULTS['PROGRESS'], min=0.0, max=1.0,
        update=apply_group_node_colors
    )

    show_grid: BoolProperty(name="Show Grid", default=True)
    extend_grid: BoolProperty(
        name="Extend Beyond Canvas",
        description="Extrapolate grid lines across the entire Node Editor viewport",
        default=False
    )
    snap_to_grid: BoolProperty(name="Snap to Grid", default=True)
    grid_size: IntProperty(name="Grid Size (px)", default=20, min=2, max=256)
    grid_color: FloatVectorProperty(
        name="Grid Color",
        subtype='COLOR',
        size=4,
        default=(0.5, 0.5, 0.5, 0.16),
        min=0.0, max=1.0
    )

    active_element_index: IntProperty(
        name="Active Element Index",
        default=0,
        update=on_active_index_update
    )

    parent_selector: EnumProperty(
        name="Parent",
        description="Assign a valid parent to the active element",
        items=get_parent_selector_items,
        get=get_parent_selector_value,
        set=set_parent_selector_value
    )

    export_filepath: StringProperty(
        name="Saved JSON Path",
        description="Last saved/loaded path for this canvas JSON",
        subtype='FILE_PATH',
        default=""
    )
    auto_save_on_play: BoolProperty(
        name="Auto-Save Canvas JSON",
        description="Refresh the saved canvas JSON when saving this Blender file",
        default=True
    )
    is_dirty: BoolProperty(name="Unsaved Changes", default=False)

class ArmoryCanvasSocket(NodeSocket):
    bl_idname = 'ArmoryCanvasSocketType'
    bl_label = 'UI Element Socket'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (0.2, 0.7, 1.0, 1.0)

# -------------------------------------------------------------------------
# 2. Base Node & All Armory2D Element Types (Unified Asset & Top Sockets)
# -------------------------------------------------------------------------
ELEMENT_TYPE_ITEMS = [
    ('3',  "Empty (3)", ""),
    ('0',  "Text (0)", ""),
    ('1',  "Image (1)", ""),
    ('2',  "Button (2)", ""),
    ('6',  "Check (6)", ""),
    ('7',  "Radio (7)", ""),
    ('10', "Text Input (10)", ""),
    ('20', "Text Area (20)", ""),
    ('11', "Key Input (11)", ""),
    ('8',  "Combo Box (8)", ""),
    ('9',  "Slider (9)", ""),
    ('13', "Rect (13)", ""),
    ('12', "Fill Rect (12)", ""),
    ('15', "Circle (15)", ""),
    ('14', "Fill Circle (14)", ""),
    ('17', "Triangle (17)", ""),
    ('16', "Fill Triangle (16)", ""),
    ('18', "RectPB (18)", ""),
    ('19', "CircularPB (19)", ""),
]

TEXT_CAPABLE_TYPES = {0, 2, 6, 7, 8, 9, 10, 11, 20}
TEXT_LAYOUT_TYPES = {0, 2, 6, 7, 10, 11, 20}
TEXT_OVERRIDE_TYPES = {2, 6, 7, 8, 9, 10, 11, 20}
TEXT_HOVER_TYPES = TEXT_OVERRIDE_TYPES - {7, 20}
INPUT_OUTLINE_TYPES = {10, 11, 20}
_BLF_FONT_CACHE = {}

def get_blf_font_id(node):
    if not node or not node.asset_name:
        return 0
    font_db = bpy.data.fonts.get(node.asset_name)
    if font_db is None:
        for fnt in bpy.data.fonts:
            if os.path.basename(bpy.path.abspath(fnt.filepath)) == node.asset_name:
                font_db = fnt
                break
    if font_db is None and node.asset_path:
        cand_p = os.path.abspath(bpy.path.abspath(os.path.join(os.path.dirname(node.id_data.export_filepath or ""), node.asset_path)))
        if os.path.isfile(cand_p):
            try:
                font_db = bpy.data.fonts.load(cand_p, check_existing=True)
            except Exception:
                pass
    if font_db is None:
        return 0

    raw_fp = (getattr(font_db, "filepath", "") or "").strip()
    if not raw_fp or raw_fp == "<builtin>":
        return 0
    abs_fp = os.path.abspath(bpy.path.abspath(raw_fp))
    if not os.path.isfile(abs_fp):
        return 0
    cached_id = _BLF_FONT_CACHE.get(abs_fp)
    if cached_id is not None and cached_id >= 0:
        return cached_id
    try:
        fid = blf.load(abs_fp)
        if fid >= 0:
            _BLF_FONT_CACHE[abs_fp] = fid
            return fid
    except Exception:
        pass
    return 0

def get_canvas_image(node):
    if node.asset_type != 'IMAGE':
        return None
    if node.asset_image is not None:
        return node.asset_image

    # Existing .blend files may lack an image pointer. Match the path first
    # so identically named files in different directories remain distinct.
    if node.asset_path:
        path = os.path.normcase(os.path.normpath(bpy.path.abspath(node.asset_path)))
        for image in bpy.data.images:
            if image.filepath and os.path.normcase(os.path.normpath(
                    bpy.path.abspath(image.filepath, library=image.library))) == path:
                return image
        return None
    return bpy.data.images.get(node.asset_name)


def update_button_image(node, context):
    node.button_image = None
    if resolve_asset_type(node.button_image_path) == 'IMAGE':
        try:
            node.button_image = bpy.data.images.load(bpy.path.abspath(node.button_image_path), check_existing=True)
        except (OSError, RuntimeError):
            pass
    mark_tree_dirty(node, context)


def apply_image_aspect_ratio(node):
    if not node or node.asset_type != 'IMAGE' or node.slice_enabled:
        return
    img = get_canvas_image(node)
    if img and len(img.size) >= 2:
        img_w, img_h = img.size[0], img.size[1]
        if img_w > 0 and img_h > 0:
            aspect = img_w / float(img_h)
            node.height = max(1, int(round(node.width / aspect)))

def on_keep_aspect_update(self, context):
    if self.keep_aspect_ratio and self.asset_type == 'IMAGE':
        apply_image_aspect_ratio(self)
    mark_tree_dirty(self, context)

def on_asset_path_update(self, context):
    for key in ('_armory_imported_asset_name', '_armory_imported_display_name',
                '_armory_imported_asset_path'):
        if key in self:
            del self[key]
    self.asset_image = None
    detected = resolve_asset_type(self.asset_path)
    self.asset_type = detected
    if self.asset_path:
        self.asset_name = os.path.basename(self.asset_path.replace('\\', '/'))
    else:
        self.asset_name = ""
    if detected == 'FONT':
        try:
            bpy.data.fonts.load(bpy.path.abspath(self.asset_path), check_existing=True)
        except Exception:
            pass
    elif detected == 'IMAGE':
        try:
            self.asset_image = bpy.data.images.load(bpy.path.abspath(self.asset_path), check_existing=True)
            if self.keep_aspect_ratio:
                apply_image_aspect_ratio(self)
        except (OSError, RuntimeError) as exc:
            print(f'Armory Canvas: unable to load image {self.asset_path}: {exc}')
    mark_tree_dirty(self, context)

def reset_step_preview(self, context):
    self.pop('_preview_step_value', None)
    mark_tree_dirty(self, context)


def reset_switch_preview(self, context):
    self.pop('_preview_switch_value', None)
    mark_tree_dirty(self, context)


def on_progress_value_update(self, context):
    # Zui progress range: minimum 0, maximum defined by Total Progress.
    maximum = max(1, int(self.progress_total))
    if self.progress_at > maximum:
        self.progress_at = maximum
    mark_tree_dirty(self, context)

def on_preview_combo_index_update(self, context):
    options = [x.strip() for x in self.preview_combo_items.splitlines() if x.strip()]
    if self.preview_combo_index >= len(options):
        self.preview_combo_index = max(0, len(options) - 1)
    if context and context.area:
        context.area.tag_redraw()

def on_preview_combo_items_update(self, context):
    on_preview_combo_index_update(self, context)

def on_progress_total_update(self, context):
    if self.progress_at > self.progress_total:
        self.progress_at = self.progress_total
    mark_tree_dirty(self, context)

CANVAS_ANCHOR_ITEMS = [
    ('0', "Top Left", ""), ('1', "Top Center", ""), ('2', "Top Right", ""),
    ('3', "Middle Left", ""), ('4', "Center", ""), ('5', "Middle Right", ""),
    ('6', "Bottom Left", ""), ('7', "Bottom Center", ""), ('8', "Bottom Right", ""),
]


STRETCH_ITEMS = (
    ('NONE', "None", "Keep the element size fixed", 0),
    ('WIDTH', "Width", "Follow the width of the parent or canvas", 1),
    ('HEIGHT', "Height", "Follow the height of the parent or canvas", 2),
    ('BOTH', "Both", "Follow the width and height of the parent or canvas", 3),
)
STRETCH_VALUES = {item[0]: item[3] for item in STRETCH_ITEMS}


def canvas_container_size(node, axis, visited, *, preview=True):
    parent = get_parent_node(node)
    if parent is not None and parent not in visited:
        return canvas_element_size(parent, axis, visited, preview=preview)
    tree = node.id_data
    if preview:
        return canvas_preview_size(tree)[0 if axis == 'width' else 1]
    return getattr(tree, 'canvas_' + axis)


def canvas_element_size(node, axis, visited=None, *, preview=True):
    default = 150 if axis == 'width' else 48
    if active_layout_parent(node) is not None:
        box = layout_child_box(node.id_data, node, preview=preview, visited=visited)
        if box is not None:
            return box[2 if axis == 'width' else 3]
    bit = 1 if axis == 'width' else 2
    if not node.get('_stretch_mode', 0) & bit:
        return clamp_canvas_size(node, 0 if axis == 'width' else 1, node.get(axis, default))
    ancestors = {node} if visited is None else visited | {node}
    container = canvas_container_size(node, axis, ancestors, preview=preview)
    return clamp_canvas_size(node, 0 if axis == 'width' else 1, container + node.get('_stretch_' + axis, 0))


def set_canvas_element_size(node, axis, size):
    node[axis] = size
    bit = 1 if axis == 'width' else 2
    if node.get('_stretch_mode', 0) & bit and active_layout_parent(node) is None:
        node['_stretch_' + axis] = size - canvas_container_size(node, axis, {node})


def set_canvas_stretch_mode(node, mode):
    if active_layout_parent(node) is not None:
        node['_stretch_mode'] = mode
        return
    width, height = node.width, node.height
    node['_stretch_mode'] = mode
    node.width, node.height = width, height


class ArmoryCanvasNodeBase(Node):
    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'ArmoryCanvasTreeType'

    default_armory_type = '12'
    node_group_category = 'SHAPES'

    z_order: IntProperty(name="Outliner Order", default=0, update=mark_canvas_order_dirty)
    children_locked: BoolProperty(name="Lock Descendants",
                                  get=lambda n: n.get('_children_locked', n.group_locked),
                                  set=lambda n,v: n.__setitem__('_children_locked', bool(v)))
    group_locked: BoolProperty(
        name="Lock Element", description="Lock this element and its descendants",
        default=False, options={'HIDDEN'},
    )
    outliner_expanded: BoolProperty(
        name="Expand Children",
        description="Expand or collapse child elements in the Outliner Hierarchy",
        default=True
    )
    armory_type: EnumProperty(name="Type", items=ELEMENT_TYPE_ITEMS, default='12', update=mark_tree_dirty)
    element_name: StringProperty(name="Name", default="Element", update=mark_tree_dirty)
    event_name: StringProperty(name="Event", default="", update=mark_tree_dirty)
    pos_x: IntProperty(name="X", default=40, update=mark_tree_dirty)
    pos_y: IntProperty(name="Y", default=40, update=mark_tree_dirty)
    width: IntProperty(
        name="Width", default=150, min=1, update=mark_tree_dirty,
        get=lambda node: canvas_element_size(node, 'width'),
        set=lambda node, size: set_canvas_element_size(node, 'width', size),
    )
    height: IntProperty(
        name="Height", default=48, min=1, update=mark_tree_dirty,
        get=lambda node: canvas_element_size(node, 'height'),
        set=lambda node, size: set_canvas_element_size(node, 'height', size),
    )
    stretch_mode: EnumProperty(
        name="Stretch", items=STRETCH_ITEMS,
        get=lambda node: node.get('_stretch_mode', 0), set=set_canvas_stretch_mode,
        update=mark_tree_dirty,
    )
    rotation: FloatProperty(name="Rotation", default=0.0, subtype='ANGLE', update=mark_tree_dirty)
    visible: BoolProperty(name="Visible", default=True, update=mark_tree_dirty)

    # Keep the old RNA identifiers so existing Blender files keep their layout settings.
    radio_layout_mode: EnumProperty(
        name="Layout Group",
        description="Arrange direct children vertically or horizontally in Outliner order",
        items=[
            ('NONE', "Manual", "Keep manual X/Y positions"),
            ('VERTICAL', "Vertical", "Stack children top-to-bottom"),
            ('HORIZONTAL', "Horizontal", "Line up children left-to-right"),
            ('GRID', "Grid", "Arrange children in cells, row by row"),
        ],
        default='NONE',
        update=mark_tree_dirty,
    )
    grid_rows: IntProperty(name="Rows", description="Minimum row count; overflow creates extra rows", min=1, max=256, default=2, update=mark_tree_dirty)
    grid_columns: IntProperty(name="Columns", min=1, max=256, default=2, update=mark_tree_dirty)
    min_size: IntVectorProperty(name="Minimum Size", size=2, min=0, default=(0, 0), update=mark_tree_dirty)
    max_size: IntVectorProperty(name="Maximum Size", description="Zero means no maximum; a minimum takes precedence", size=2, min=0, default=(0, 0), update=mark_tree_dirty)
    wrap_text: BoolProperty(name="Word Wrap", default=False, update=mark_tree_dirty)
    wrap_width: IntProperty(name="Max Line Width", description="Pixels; zero uses the element width", min=0, default=0, update=mark_tree_dirty)
    disabled: BoolProperty(name="Disabled", description="Draw the control in grey and prevent interaction", default=False, update=mark_tree_dirty)
    slice_enabled: BoolProperty(name="9-Slice", get=lambda n: n.get('_slice_enabled', any(n.slice_borders)),
                                set=lambda n,v: n.__setitem__('_slice_enabled', bool(v)), update=mark_tree_dirty)
    slice_borders: IntVectorProperty(name="9-Slice Borders", description="Source texture pixels, left/right/top/bottom; all zero disables slicing", size=4, min=0, default=(0, 0, 0, 0), update=mark_tree_dirty)
    corner_radius: IntProperty(name="Corner Radius", min=0, default=0, update=mark_tree_dirty)
    shadow_enabled: BoolProperty(name="Drop Shadow", default=False, update=mark_tree_dirty)
    shadow_offset: IntVectorProperty(name="Shadow Offset", description="X right, Y down", size=2, default=(4, 4), update=mark_tree_dirty)
    shadow_color: FloatVectorProperty(name="Shadow Color", size=4, subtype='COLOR', min=0, max=1, default=(0, 0, 0, 0.35), update=mark_tree_dirty)
    mask_children: BoolProperty(name="Mask Children", description="Clip descendants to this element's rectangular bounds", update=mark_tree_dirty)
    scroll_content: IntVectorProperty(name="Content Size", size=2, min=0, default=(0, 0), description="Minimum scrollable width and height in canvas pixels", update=mark_tree_dirty)
    scroll_bar_size: IntVectorProperty(name="Scrollbar Size", size=2, default=(8, 8), min=1, max=1024, update=mark_tree_dirty)
    scroll_bar_color: FloatVectorProperty(name="Thumb Color", subtype='COLOR', size=4, min=0, max=1,
                                         default=(0.50, 0.61, 0.67, 0.98), update=mark_tree_dirty)
    scroll_track_color: FloatVectorProperty(name="Track Color", subtype='COLOR', size=4, min=0, max=1,
                                           default=(0.04, 0.06, 0.08, 0.7), update=mark_tree_dirty)
    text_padding_vertical: IntVectorProperty(name="Vertical Padding", size=2, default=(0, 0),
                                              min=0, update=mark_tree_dirty)
    line_enabled: BoolProperty(name="Line", default=False, update=mark_tree_dirty)
    line_color: FloatVectorProperty(name="Color", subtype='COLOR', size=4, min=0, max=1,
                                    default=(0, 0, 0, 1), update=mark_tree_dirty)
    line_thickness: FloatProperty(name="Thickness", default=1, min=0.1, max=256, update=mark_tree_dirty)
    line_position: EnumProperty(name="Position", items=[
        ('OUTSIDE', "Outside", ""), ('INSIDE', "Inside", ""), ('CENTER', "Center", ""),
    ], default='CENTER', update=mark_tree_dirty)
    scroll_axes: BoolVectorProperty(name="Scroll Axes", description="Enable clipping and scrolling along either axis", size=2, default=(False, False), update=mark_tree_dirty)
    preview_scroll: FloatVectorProperty(name="Preview Scroll", description="Current preview offset in canvas pixels", size=2, min=0, default=(0, 0), update=update_preview_scroll)
    control_variant: EnumProperty(name="Control", items=[('DEFAULT', "Default", ""), ('STEPPER', "Numeric Stepper", ""), ('SWITCH', "Toggle Switch", "")], default='DEFAULT', update=mark_tree_dirty)
    step_min: FloatProperty(name="Minimum", default=0, update=mark_tree_dirty)
    step_max: FloatProperty(name="Maximum", default=100, update=mark_tree_dirty)
    step_amount: FloatProperty(name="Step", min=0.000001, default=1, update=mark_tree_dirty)
    step_initial: FloatProperty(name="Initial Value", default=0, update=reset_step_preview)
    switch_initial: BoolProperty(name="Initially On", default=False, update=reset_switch_preview)

    radio_spacing: IntProperty(
        name="Spacing",
        description="Gap in pixels between layout children",
        default=8, min=0, max=512,
        update=mark_tree_dirty,
    )
    layout_padding: IntVectorProperty(
        name="Padding", description="Left, right, top and bottom padding in pixels",
        size=4, default=(0, 0, 0, 0), min=0, update=mark_tree_dirty,
    )
    layout_child_alignment: EnumProperty(
        name="Child Alignment", items=CANVAS_ANCHOR_ITEMS, default='0', update=mark_tree_dirty,
    )
    layout_use_child_scale: BoolProperty(name="Use Child Scale",
        description="Keep each child's authored size and reserve space for its non-visual-only offset scale",
        default=False, update=mark_tree_dirty)
    offset_position: FloatVectorProperty(name="Position Offset", size=2, default=(0, 0), update=mark_tree_dirty)
    offset_rotation: FloatProperty(name="Rotation Offset", subtype='ANGLE', default=0, update=mark_tree_dirty)
    offset_scale: FloatProperty(name="Legacy Scale Offset", default=1, min=0.01, max=100,
                                options={'HIDDEN', 'ANIMATABLE'}, update=mark_tree_dirty)
    offset_scale_xy: FloatVectorProperty(name="Scale", size=2, default=(1, 1), min=0.01, max=100,
        get=lambda node: node.get("offset_scale_xy", (node.offset_scale, node.offset_scale)),
        set=lambda node, value: node.__setitem__("offset_scale_xy", list(value)), update=mark_tree_dirty)
    offset_transform_expanded: BoolProperty(name="Transform Offset", default=True, options=set())
    offset_visual_only: BoolProperty(name="Visual Only", default=True,
        description="Keep the original interaction bounds and layout footprint", update=mark_tree_dirty)
    layout_control_size: BoolVectorProperty(
        name="Control Child Size", size=2, default=(False, False), update=mark_tree_dirty,
        description="Let the group resize children on this axis. Disable to keep each child's authored size",
    )
    layout_force_expand: BoolVectorProperty(
        name="Child Force Expand", size=2, default=(False, False), update=mark_tree_dirty,
    )
    layout_proportional: BoolProperty(
        name="Equal Distribution",
        description="Give children equal slots. Control Child Size determines which axes fill the slot",
        default=False, update=mark_tree_dirty,
    )
    layout_reverse: BoolProperty(name="Reverse Order", default=False, update=mark_tree_dirty)
    ignore_parent_layout: BoolProperty(name="Ignore Parent Layout", default=False, update=mark_tree_dirty)
    radio_simulate: BoolProperty(
        name="Simulate Radio Group",
        description="Click child Radio elements on the canvas to preview exclusive selection (preview only)",
        default=False,
    )

    text_content: StringProperty(name="Text", default="", update=mark_tree_dirty)
    font_size: IntProperty(
        name="Font Size",
        description="Fixed font size in pixels in the Blender preview",
        default=24, min=4, max=256,
        update=mark_tree_dirty
    )
    font_mode: EnumProperty(
        name="Font Size Mode",
        description="Choose the font size in Blender and in Armory with the Canvas font-mode patch",
        items=[
            ('THEME', "Use Theme", "Use the active Zui theme's font size", 0),
            ('FIXED', "Fixed Size", "Use a font size in pixels", 1),
            ('HEIGHT', "Scale with Height", "Use 72% of the element height", 2),
        ],
        get=get_font_mode, set=set_font_mode,
    )
    preview_font_override: BoolProperty(name='Custom Preview Font Size', description='Blender preview only; Armory Canvas JSON has no verified per-element font size field', default=False, update=mark_tree_dirty)
    multiline: BoolProperty(name="Multiline", default=False, update=mark_tree_dirty)
    multiline_text: StringProperty(
        name="Text", description=r"Use \n for a line break and \\ for a literal backslash",
        get=get_multiline_text, set=set_multiline_text,
    )
    use_local_padding: BoolProperty(
        name="Custom Padding", default=False, update=mark_tree_dirty,
    )
    text_offset: IntProperty(
        name="Fixed Padding",
        description="Horizontal text inset in pixels for left or right alignment",
        default=8, min=0, max=256,
        update=mark_tree_dirty
    )

    keep_aspect_ratio: BoolProperty(
        name="Keep Aspect Ratio",
        description="Maintain image aspect ratio when scaling and rendering instead of distorting",
        default=False,
        update=on_keep_aspect_update
    )

    # Unified Asset Properties
    button_image_path: StringProperty(name="Background Image", subtype='FILE_PATH', update=update_button_image)
    button_image: PointerProperty(type=bpy.types.Image, options={'HIDDEN'})
    asset_image: PointerProperty(type=bpy.types.Image, options={'HIDDEN'})
    asset_name: StringProperty(name="Asset Name", default="", update=mark_tree_dirty)
    asset_path: StringProperty(
        name="Asset Path",
        description="Unified path to font or image file",
        subtype='NONE',
        default="../../Sources/",
        update=on_asset_path_update
    )
    asset_type: EnumProperty(
        name="Asset Type",
        items=[
            ('NONE', "None", "No asset assigned"),
            ('IMAGE', "Image", "Image texture or sprite"),
            ('FONT', "Font", "Typography font file"),
        ],
        default='NONE',
        update=mark_tree_dirty
    )

    # Preview-only interaction values: do not add unsupported fields to Armory JSON.
    preview_slider_value: FloatProperty(name="Preview Slider Value", default=0.5, min=0.0, max=1.0)
    preview_checked: BoolProperty(name="Preview Checked", default=False)
    radio_style: EnumProperty(name="Radio Style", items=[
        ('RECT', "Rect", "Fill the radio element with a rectangle"),
        ('DOT', "Dot", "Use a circular radio indicator beside the label"),
    ], default='RECT', update=mark_tree_dirty)
    preview_radio_selected: BoolProperty(name="Preview Radio Selected", default=False)
    preview_combo_items: StringProperty(name="Preview Combo Options", default="Option 1\nOption 2\nOption 3", description="One option per line; preview only", update=on_preview_combo_items_update)
    preview_combo_open: BoolProperty(name="Show Dropdown", default=False, description="Display the combo options in the visual canvas; preview only")
    preview_combo_index: IntProperty(name="Preview Selected Index", default=0, min=0, update=on_preview_combo_index_update)
    preview_key_hover: BoolProperty(name="Preview Hover", default=False)
    preview_input_text: StringProperty(name="Preview Input Text", default="")

    progress_at: IntProperty(
        name="Progress",
        description="Current progress value",
        default=50, min=0, max=1000000,
        update=on_progress_value_update
    )
    progress_total: IntProperty(
        name="Total Progress",
        description="Maximum progress value",
        default=100, min=1, max=1000000,
        update=on_progress_total_update
    )
    strength: FloatProperty(
        name="Thickness",
        description="Line or ring stroke thickness in pixels",
        default=2.0, min=0.5, max=128.0,
        update=mark_tree_dirty
    )

    alignment: EnumProperty(
        name="Text Align",
        items=[('0', "Left", ""), ('1', "Center", ""), ('2', "Right", "")],
        default='0',
        update=mark_tree_dirty
    )
    text_vertical_align: EnumProperty(
        name="Vertical Text Align",
        items=[('TOP', "Top", ""), ('MIDDLE', "Middle", ""), ('BOTTOM', "Bottom", "")],
        default='MIDDLE', update=mark_tree_dirty,
    )
    pivot: FloatVectorProperty(name="Pivot", size=2, default=(.5, .5), min=0, max=1, update=mark_tree_dirty)
    anchor: EnumProperty(
        name="Anchor",
        items=CANVAS_ANCHOR_ITEMS,
        default='0',
        update=mark_tree_dirty
    )

    use_end_color: BoolProperty(name="Base Button Color", update=mark_tree_dirty)
    end_color: FloatVectorProperty(name="Base Button Color", subtype="COLOR", size=4, default=(0.22, 0.24, 0.28, 1), min=0, max=1, update=mark_tree_dirty)
    use_element_color: BoolProperty(name="Base / BG Color", default=False, update=mark_tree_dirty)
    element_color: FloatVectorProperty(
        name="Color", subtype='COLOR', size=4,
        default=(0.22, 0.24, 0.28, 1.0), min=0.0, max=1.0, update=mark_tree_dirty
    )

    use_progress_color: BoolProperty(name="Progress Color", default=False, update=mark_tree_dirty)
    color_progress: FloatVectorProperty(
        name="Progress Color", subtype='COLOR', size=4,
        default=(0.15, 0.65, 0.95, 1.0), min=0.0, max=1.0, update=mark_tree_dirty
    )

    use_custom_button_states: BoolProperty(name="Button Text/Hover/Press", default=False, update=mark_tree_dirty)
    color_text: FloatVectorProperty(
        name="Text Color", subtype='COLOR', size=4,
        default=(1.0, 0.95, 0.8, 1.0), min=0.0, max=1.0, update=mark_tree_dirty
    )
    color_hover: FloatVectorProperty(
        name="Hover Color", subtype='COLOR', size=4,
        default=(0.7, 0.6, 0.0, 1.0), min=0.0, max=1.0, update=mark_tree_dirty
    )
    color_press: FloatVectorProperty(
        name="Press Color", subtype='COLOR', size=4,
        default=(0.0, 0.4, 0.95, 1.0), min=0.0, max=1.0, update=mark_tree_dirty
    )

    def apply_default_group_color(self, ntree):
        if not isinstance(ntree, ArmoryCanvasTree):
            return
        group_map = {
            'BASIC':    ntree.node_color_basic,
            'BUTTONS':  ntree.node_color_buttons,
            'INPUTS':   ntree.node_color_inputs,
            'SHAPES':   ntree.node_color_shapes,
            'PROGRESS': ntree.node_color_progress,
        }
        if getattr(ntree, "color_code_nodes", True) and self.node_group_category in group_map:
            self.use_custom_color = True
            self.color = tuple(group_map[self.node_group_category])

    def init(self, context):
        self.armory_type = self.default_armory_type
        self.element_name = self.bl_label
        self.hide = True
        self.inputs.new('ArmoryCanvasSocketType', "Children").link_limit = 4095
        self.outputs.new('ArmoryCanvasSocketType', "Parent").link_limit = 1

        ntree = self.id_data
        if isinstance(ntree, ArmoryCanvasTree):
            self.apply_default_group_color(ntree)
            for n in ntree.nodes:
                if isinstance(n, ArmoryCanvasNodeBase) and n != self:
                    n.z_order += 1
            self.z_order = 0
            ntree.nodes.active = self
            for other in ntree.nodes:
                other.select = (other == self)
            sync_active_element_index(ntree, self)

        mark_tree_dirty(self, context)

    def free(self):
        ntree = self.id_data
        mark_tree_dirty(self, bpy.context)
        if isinstance(ntree, ArmoryCanvasTree):
            remaining = [n for n in get_ordered_canvas_nodes(ntree) if n != self]
            if remaining:
                next_active = ntree.nodes.active if (ntree.nodes.active in remaining) else remaining[0]
                ntree.nodes.active = next_active
                sync_active_element_index(ntree, next_active)
            else:
                ntree.active_element_index = 0

    def is_effectively_visible(self, parent_of=get_parent_node):
        if not self.visible:
            return False
        visited = {self}
        curr = parent_of(self)
        while curr is not None and curr not in visited:
            if not curr.visible:
                return False
            visited.add(curr)
            curr = parent_of(curr)
        return True

    def draw_buttons_ext(self, context, layout):
        self.draw_element_properties(context, layout)

    def draw_element_properties(self, context, layout):
        layout.enabled = not element_selection_locked(self)
        ntree = self.id_data

        main = layout.box()
        main.label(text="Main Properties")
        main.prop(self, "armory_type")
        row = main.row(align=True)
        row.prop(self, "element_name")
        row.prop(self, "visible", text="", icon='HIDE_OFF' if self.visible else 'HIDE_ON')
        parent = get_parent_node(self)
        if ntree.nodes.active == self:
            main.prop(ntree, "parent_selector", text="Parent")
        else:
            main.label(text=f"Parent: {parent.element_name if parent else 'Canvas Root'}")

        script = layout.box()
        script.label(text="Script")
        script.prop(self, "event_name")

        position = layout.box()
        position.label(text="Position")
        position.context_pointer_set("canvas_element", self)
        manual_position = position.column(align=True)
        manual_position.enabled = not self.visible or active_layout_parent(self) is None
        manual_position.popover(panel="NODE_PT_armory_canvas_anchor_presets", text="Anchor Presets")
        row = manual_position.row(align=True)
        row.prop(self, "pos_x")
        row.prop(self, "pos_y")
        position.prop(self, "rotation")
        draw_canvas_pair(position, self, "pivot", "Pivot", ("X", "Y"))
        offset = position.box()
        offset.prop(self, "offset_transform_expanded", text="Transform Offset", emboss=False,
                    icon='TRIA_DOWN' if self.offset_transform_expanded else 'TRIA_RIGHT')
        if self.offset_transform_expanded:
            draw_canvas_pair(offset, self, "offset_position", "Position", ("X", "Y"))
            offset.prop(self, "offset_rotation", text="Rotation")
            draw_canvas_pair(offset, self, "offset_scale_xy", "Scale", ("X", "Y"))
            offset.prop(self, "offset_visual_only")

        sizing = layout.box()
        sizing.label(text="Layout")
        dimensions = sizing.row(align=True)
        for axis, editable in zip(("width", "height"), canvas_resize_axes(self)):
            field = dimensions.row(align=True)
            field.enabled = editable
            field.prop(self, axis)
        parent = get_parent_node(self)

        draw_canvas_pair(sizing, self, "min_size", "Minimum Size", ('Width', 'Height'))
        draw_canvas_pair(sizing, self, "max_size", "Maximum Size", ('Width', 'Height'))
        if self.stretch_mode != 'NONE' and active_layout_parent(self) is None:
            sizing.context_pointer_set("canvas_element", self)
            sizing.operator("node.armory_canvas_fit_stretch")
        if parent is not None and parent.radio_layout_mode != 'NONE':
            sizing.prop(self, "ignore_parent_layout")

        t_int = int(self.armory_type)
        specific = layout.box()
        specific.label(text="Properties")
        if t_int in {3, 12, 13}:
            group = specific
            group.prop(self, "radio_layout_mode")
            axes = group.row(align=True)
            for i, label in enumerate(("Horizontal Scroll", "Vertical Scroll")):
                axes.prop(self, "scroll_axes", index=i, text=label, toggle=True)
            if any(self.scroll_axes):
                draw_canvas_pair(group, self, "scroll_content", "Content Size", ('Width', 'Height'))
                draw_canvas_pair(group, self, "preview_scroll", "Preview Offset", ('X', 'Y'))
                scroll = group.box()
                draw_canvas_pair(scroll, self, "scroll_bar_size", "Scrollbar", ("Vertical Width", "Horizontal Height"))
                scroll.prop(self, "scroll_bar_color")
                scroll.prop(self, "scroll_track_color")
            if self.radio_layout_mode != 'NONE':
                group.prop(self, "radio_spacing")
                for index, label in enumerate(("Left", "Right", "Top", "Bottom")):
                    group.prop(self, "layout_padding", index=index, text=label)
                if self.radio_layout_mode == 'GRID':
                    grid = group.row(align=True)
                    grid.prop(self, "grid_rows")
                    grid.prop(self, "grid_columns")
                    group.prop(self, "layout_child_alignment")
                else:
                    group.prop(self, "layout_use_child_scale")
                    proportional = group.row()
                    proportional.enabled = not self.layout_use_child_scale
                    proportional.prop(self, "layout_proportional")
                    group.prop(self, "layout_child_alignment")
                    sizing_options = group.column()
                    sizing_options.enabled = not self.layout_use_child_scale
                    for property_name, label in (
                        ("layout_control_size", "Control Child Size"), ("layout_force_expand", "Child Force Expand")
                    ):
                        options = sizing_options.column(align=True)
                        if property_name == "layout_force_expand":
                            options.enabled = not self.layout_proportional
                        options.label(text=label)
                        axis_row = options.row(align=True)
                        axis_row.prop(self, property_name, index=0, text="Width")
                        axis_row.prop(self, property_name, index=1, text="Height")
                group.prop(self, "layout_reverse")
        if t_int == 3:
            style = layout.box()
            style.label(text="Style")
            style.prop(self, "mask_children", text="Mask Children")
            self.draw_line_settings(style)
            return
        if t_int in CANVAS_DISABLED_TYPES:
            specific.prop(self, "disabled")
        if t_int in {2, 6}:
            row = specific.row(align=True)
            variant = 'STEPPER' if t_int == 2 else 'SWITCH'
            row.prop_enum(self, "control_variant", 'DEFAULT')
            row.prop_enum(self, "control_variant", variant)
            if self.control_variant == variant:
                for prop in ('step_min', 'step_max', 'step_amount', 'step_initial') if t_int == 2 else ('switch_initial',):
                    specific.prop(self, prop)
        if t_int in {18, 19}:
            col_prog = specific.column(align=True)
            col_prog.prop(self, "progress_total", text="Maximum (Minimum = 0)")
            col_prog.prop(self, "progress_at", text="Progress")
            col_prog.label(text=f"Range: 0 - {self.progress_total}  |  {min(self.progress_at, self.progress_total) / max(1, self.progress_total):.0%}")

        if t_int == 7:
            specific.prop(self, "radio_style", expand=True)
        if t_int in TEXT_CAPABLE_TYPES:
            if t_int == 0:
                specific.prop(self, "multiline")
                specific.prop(self, "wrap_text")
                if self.wrap_text:
                    specific.prop(self, "wrap_width")
            if t_int == 0 and (self.multiline or self.wrap_text):
                specific.prop(self, "multiline_text")
                specific.label(text=r"Type \n to start a new line")
            else:
                specific.prop(self, "text_content")

            specific = layout.box()
            specific.label(text="Typography")
            self.draw_asset_settings(specific, "Font")
            specific.prop(self, "font_mode")
            if self.font_mode == 'FIXED':
                specific.prop(self, "font_size")
            else:
                specific.label(text=f"Preview size: {canvas_node_font_size(self, ntree)} px")
            if t_int in TEXT_LAYOUT_TYPES:
                specific.prop(self, "use_local_padding", text="Custom Padding")
                if self.use_local_padding:
                    specific.prop(self, "text_offset")
                else:
                    specific.label(text=f"Theme padding: {canvas_text_padding(self)} px")
            specific.prop(self, "alignment")
            if t_int in TEXT_LAYOUT_TYPES:
                specific.prop(self, "text_vertical_align")
                draw_canvas_pair(specific, self, "text_padding_vertical", "Vertical Padding", ("Top", "Bottom"))
        else:
            if t_int == 1:
                self.draw_asset_settings(specific, "Image")
                self.draw_slice_settings(specific)
                if not self.slice_enabled:
                    specific.prop(self, "keep_aspect_ratio")

        layout = layout.box()
        layout.label(text="Style")
        if t_int == 2 and self.control_variant != 'STEPPER':
            row = layout.row(align=True)
            row.context_pointer_set("node", self)
            row.prop(self, "button_image_path")
            row.operator("ui_canvas.select_asset", text="", icon='FILEBROWSER').button_background = True
            if self.button_image_path:
                self.draw_slice_settings(layout)
        layout.prop(self, "mask_children")
        if t_int in {13, 15, 17, 18, 19}:
            layout.prop(self, "strength")
        self.draw_line_settings(layout)
        if t_int in CANVAS_RADIUS_TYPES:
            layout.prop(self, "corner_radius")
        if t_int in CANVAS_STYLE_TYPES:
            shadow = layout.column(align=True)
            shadow.prop(self, "shadow_enabled", text="Drop Shadow")
            if self.shadow_enabled:
                draw_canvas_pair(shadow, self, "shadow_offset", "Shadow Offset",
                                 ("Left / Right", "Top / Down"))
                shadow.prop(self, "shadow_color")
        if t_int in {18, 19}:
            col_pcol = layout.column(align=True)
            col_pcol.prop(self, "use_progress_color", text="Progress Color")
            if self.use_progress_color:
                col_pcol.prop(self, "color_progress", text="Progress Color")

        if t_int != 3:
            special = t_int in {6, 8, 9, 18, 19} | INPUT_OUTLINE_TYPES
            bg_label = "Text Color" if t_int == 0 else ("Background" if special else "Color")
            layout.prop(self, "use_element_color", text="Background" if special else "Base Color")
            if self.use_element_color:
                if t_int == 2:
                    self.draw_button_color(layout, "element_color", "Normal Tint" if self.button_image_path else "Color")
                else:
                    layout.prop(self, "element_color", text=bg_label)

        variant_buttons = t_int == 2 and self.control_variant == 'STEPPER'
        if variant_buttons:
            layout.prop(self, "use_end_color", text="Base Button Color")
            if self.use_end_color:
                self.draw_button_color(layout, "end_color", "Base Button Color")
        if t_int in TEXT_OVERRIDE_TYPES:
            label = "Text / Hover" if t_int in TEXT_HOVER_TYPES else "Text Color"
            layout.prop(self, "use_custom_button_states", text="Text / Hover / Press" if variant_buttons else label)
            if self.use_custom_button_states:
                if t_int == 2:
                    for prop, label in (("color_text", "Text Color"), ("color_hover", "On Hover"),
                                        ("color_press", "On Press")):
                        self.draw_button_color(layout, prop, label)
                else:
                    layout.prop(self, "color_text", text="Text Color")
                    if t_int in TEXT_HOVER_TYPES:
                        layout.prop(self, "color_hover", text="On Hover")

    def draw_slice_settings(self, layout):
        layout.prop(self, "slice_enabled", toggle=True)
        if self.slice_enabled:
            draw_canvas_pair(layout, self, "slice_borders", "Source Borders (px)", ("Left", "Right"))
            draw_canvas_pair(layout, self, "slice_borders", "", ("Top", "Bottom"), first_index=2)

    def draw_button_color(self, layout, prop, label):
        row = layout.split(factor=0.5, align=True)
        row.label(text=label)
        row.prop(self, prop, text="")

    def draw_line_settings(self, layout):
        layout.prop(self, "line_enabled")
        if self.line_enabled:
            line = layout.column(align=True)
            line.prop(self, "line_color")
            line.prop(self, "line_thickness")
            line.prop(self, "line_position")

    def draw_asset_settings(self, layout, label):
        row = layout.row(align=True)
        row.prop(self, "asset_path", text=label)
        row.operator("ui_canvas.select_asset", text="", icon='FILEBROWSER')
        layout.prop(self, "asset_name", text=f"{label} Name")

    def get_anchor_origin(self, ntree, visited=None, *, anchor=None):
        if visited is None:
            visited = set()

        parent_node = get_parent_node(self)
        if parent_node is not None and parent_node not in visited:
            visited.add(self)
            base_x, base_y = parent_node.get_world_pos(ntree, visited)
            scroll_x, scroll_y = canvas_scroll_offset(parent_node)
            base_x -= scroll_x
            base_y -= scroll_y
            cont_w = int(parent_node.width)
            cont_h = int(parent_node.height)
        else:
            base_x, base_y = 0, 0
            cont_w, cont_h = canvas_preview_size(ntree) if ntree else (1024, 1024)

        if active_layout_parent(self) is not None:
            return base_x, base_y
        anc = int(self.anchor if anchor is None else anchor)
        col = anc % 3
        row = anc // 3

        w = int(self.width)
        h = int(self.height)

        if col == 0:
            off_x = 0
        elif col == 1:
            off_x = int(round((cont_w - w) * 0.5))
        else:
            off_x = cont_w - w

        if row == 0:
            off_y = 0
        elif row == 1:
            off_y = int(round((cont_h - h) * 0.5))
        else:
            off_y = cont_h - h

        return base_x + off_x, base_y + off_y

    def get_world_pos(self, ntree, visited=None):
        ancestors = {self} if visited is None else visited | {self}
        parent = get_parent_node(self)
        if parent is not None and parent not in ancestors:
            local = layout_child_position(ntree, self)
            if local is not None:
                x, y = parent.get_world_pos(ntree, ancestors)
                scroll_x, scroll_y = canvas_scroll_offset(parent)
                return x + local[0] - scroll_x, y + local[1] - scroll_y
        ax, ay = self.get_anchor_origin(ntree, visited)
        return int(ax + self.pos_x), int(ay + self.pos_y)

    def to_armory_dict(self, elem_id, parent_id):
        t_int = int(self.armory_type)
        data = {
            "id": elem_id,
            "type": t_int,
            "name": self.element_name,
            "event": self.event_name,
            "x": int(self.pos_x),
            "y": int(self.pos_y),
            "width": canvas_element_size(self, 'width', preview=False),
            "height": canvas_element_size(self, 'height', preview=False),
            "rotation": self.rotation,
            "_editor_rotation_unit": "radians",
            "text": self.text_content,
            "asset": canvas_runtime_asset_name(self),
            "progress_at": int(self.progress_at),
            "progress_total": int(self.progress_total),
            "strength": clean_number(self.strength),
            "alignment": int(self.alignment),
            "anchor": int(self.anchor),
            "parent": parent_id,
            "children": [],
            "visible": bool(self.visible),
        }

        local = layout_child_position(self.id_data, self, preview=False) if parent_id is not None else None
        if local is not None:
            data["_editor_layout_manual"] = [self.pos_x, self.pos_y, int(self.anchor)]
            data["x"], data["y"] = local
            data["anchor"] = 0
        if self.radio_layout_mode != 'NONE' or self.radio_simulate or self.ignore_parent_layout:
            data["_editor_layout_group"] = {
                "mode": self.radio_layout_mode, "spacing": self.radio_spacing,
                "padding": list(self.layout_padding), "alignment": self.layout_child_alignment,
                "reverse": self.layout_reverse, "radio": self.radio_simulate,
                "ignore": self.ignore_parent_layout,
                "proportional": self.layout_proportional, "use_child_scale": self.layout_use_child_scale,
                "grid_rows": self.grid_rows, "grid_columns": self.grid_columns,
                "control_size": list(self.layout_control_size), "force_expand": list(self.layout_force_expand),
            }

        if local is not None:
            data['_editor_layout_preferred_size'] = [self.get('width', 150), self.get('height', 48)]
        if self.stretch_mode != 'NONE':
            key = '_editor_layout_stretch' if local is not None else '_editor_stretch'
            data[key] = {
                'mode': self.stretch_mode,
                'offsets': [self.get('_stretch_width', 0), self.get('_stretch_height', 0)],
            }

        if t_int in TEXT_CAPABLE_TYPES:
            if self.asset_name:
                data["font"] = self.asset_name
            data["font_mode"] = self.font_mode
            if self.font_mode == 'FIXED':
                data["font_size"] = self.font_size
            if t_int == 0:
                data["multiline"] = self.multiline or self.wrap_text
                text = normalize_text_lines(self.text_content)
                data["text"] = text if self.multiline or self.wrap_text else text.replace("\n", " ")
        if t_int in TEXT_LAYOUT_TYPES:
            if t_int != 0 or self.text_vertical_align != 'MIDDLE':
                data["text_vertical_align"] = self.text_vertical_align
            if self.use_local_padding:
                data["text_offset"] = self.text_offset

        if self.use_element_color and t_int != 3:
            argb_col = rgba_to_argb_int(self.element_color)
            data["color"] = argb_col
            if t_int == 0:
                data["color_text"] = argb_col

        if t_int in {18, 19} and self.use_progress_color:
            data["color_progress"] = rgba_to_argb_int(self.color_progress)

        if t_int in TEXT_OVERRIDE_TYPES and self.use_custom_button_states:
            data["color_text"] = rgba_to_argb_int(self.color_text)
            if t_int in TEXT_HOVER_TYPES:
                data["color_hover"] = rgba_to_argb_int(self.color_hover)
            if t_int == 2:
                data["color_press"] = rgba_to_argb_int(self.color_press)

        features = export_canvas_features(self, t_int)
        if features:
            data['_editor_features'] = features

        return data

# --- Node Subclasses ---
class ArmoryCanvasEmptyNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasEmptyNode'
    bl_label = 'Empty'
    default_armory_type = '3'
    node_group_category = 'BASIC'

class ArmoryCanvasTextNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasTextNode'
    bl_label = 'Text'
    default_armory_type = '0'
    node_group_category = 'BASIC'

class ArmoryCanvasImageNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasImageNode'
    bl_label = 'Image'
    default_armory_type = '1'
    node_group_category = 'BASIC'

class ArmoryCanvasButtonNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasButtonNode'
    bl_label = 'Button'
    default_armory_type = '2'
    node_group_category = 'BUTTONS'

class ArmoryCanvasCheckNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasCheckNode'
    bl_label = 'Check'
    default_armory_type = '6'
    node_group_category = 'BUTTONS'

class ArmoryCanvasRadioNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasRadioNode'
    bl_label = 'Radio'
    default_armory_type = '7'
    node_group_category = 'BUTTONS'

class ArmoryCanvasTextInputNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasTextInputNode'
    bl_label = 'Text Input'
    default_armory_type = '10'
    node_group_category = 'INPUTS'

class ArmoryCanvasTextAreaNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasTextAreaNode'
    bl_label = 'Text Area'
    default_armory_type = '20'
    node_group_category = 'INPUTS'

class ArmoryCanvasKeyInputNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasKeyInputNode'
    bl_label = 'Key Input'
    default_armory_type = '11'
    node_group_category = 'INPUTS'

class ArmoryCanvasComboBoxNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasComboBoxNode'
    bl_label = 'Combo Box'
    default_armory_type = '8'
    node_group_category = 'INPUTS'

class ArmoryCanvasSliderNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasSliderNode'
    bl_label = 'Slider'
    default_armory_type = '9'
    node_group_category = 'INPUTS'

class ArmoryCanvasRectNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasRectNode'
    bl_label = 'Rect'
    default_armory_type = '13'
    node_group_category = 'SHAPES'

class ArmoryCanvasFilledRectNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasFilledRectNode'
    bl_label = 'Fill Rect'
    default_armory_type = '12'
    node_group_category = 'SHAPES'

class ArmoryCanvasCircleNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasCircleNode'
    bl_label = 'Circle'
    default_armory_type = '15'
    node_group_category = 'SHAPES'

class ArmoryCanvasFilledCircleNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasFilledCircleNode'
    bl_label = 'Fill Circle'
    default_armory_type = '14'
    node_group_category = 'SHAPES'

class ArmoryCanvasTriangleNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasTriangleNode'
    bl_label = 'Triangle'
    default_armory_type = '17'
    node_group_category = 'SHAPES'

class ArmoryCanvasFilledTriangleNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasFilledTriangleNode'
    bl_label = 'Fill Triangle'
    default_armory_type = '16'
    node_group_category = 'SHAPES'

class ArmoryCanvasProgressNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasProgressNode'
    bl_label = 'RectPB'
    default_armory_type = '18'
    node_group_category = 'PROGRESS'

    def init(self, context):
        super().init(context)
        self.width = 200
        self.height = 32
        self.progress_at = 50
        self.progress_total = 100

class ArmoryCanvasCProgressNode(ArmoryCanvasNodeBase):
    bl_idname = 'ArmoryCanvasCProgressNode'
    bl_label = 'CircularPB'
    default_armory_type = '19'
    node_group_category = 'PROGRESS'

    def init(self, context):
        super().init(context)
        self.width = 120
        self.height = 120
        self.strength = 8.0
        self.progress_at = 65
        self.progress_total = 100

# -------------------------------------------------------------------------
# 3. Coordinate & Shape Geometry Helpers
# -------------------------------------------------------------------------
def transform_point(lx, ly, center, cos_a, sin_a):
    return (
        lx * cos_a - ly * sin_a + center.x,
        lx * sin_a + ly * cos_a + center.y,
    )

def get_element_corners_view(node, ntree):
    cache = _canvas_draw_cache
    if cache is not None and cache['tree'] == ntree:
        corners = cache['corners']
        if node not in corners:
            corners[node] = calculate_element_corners(node, ntree)
        return corners[node]
    return calculate_element_corners(node, ntree)


def canvas_element_pose(node, tree):
    cache = _canvas_draw_cache
    poses = cache.setdefault('poses', {}) if cache is not None and cache['tree'] == tree else {}
    if node in poses:
        return poses[node]
    top = canvas_view_top(tree)

    def center_of(element):
        x, y = element.get_world_pos(tree)
        return x + element.width / 2, top - y - element.height / 2

    def pivot_of(element, center):
        x, y = element.pivot
        return Vector((center[0] + (x - .5) * element.width,
                       center[1] + (y - .5) * element.height))

    def rotate(point, pivot, angle):
        return transform_point(point[0] - pivot.x, point[1] - pivot.y, pivot,
                               math.cos(angle), math.sin(angle))

    center = center_of(node)
    angle = -node.rotation
    if angle:
        center = rotate(center, pivot_of(node, center), angle)
    visited = {node}
    parent = get_parent_node(node)
    while parent is not None and parent not in visited:
        visited.add(parent)
        if parent.rotation:
            rotation = -parent.rotation
            center = rotate(center, pivot_of(parent, center_of(parent)), rotation)
            angle += rotation
        parent = get_parent_node(parent)
    poses[node] = Vector(center), angle
    return poses[node]


def canvas_pivot_view(node, tree):
    center, angle = canvas_element_pose(node, tree)
    return transform_point((node.pivot[0] - .5) * node.width,
                           (node.pivot[1] - .5) * node.height,
                           center, math.cos(angle), math.sin(angle))


def calculate_element_corners(node, ntree):
    w, h = float(node.width), float(node.height)
    center, angle = canvas_element_pose(node, ntree)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    local_corners = [
        (-w * 0.5, -h * 0.5),
        ( w * 0.5, -h * 0.5),
        ( w * 0.5,  h * 0.5),
        (-w * 0.5,  h * 0.5),
    ]
    world_corners = [transform_point(lx, ly, center, cos_a, sin_a) for lx, ly in local_corners]
    rot_handle = transform_point(0.0, h * 0.5 + 32.0, center, cos_a, sin_a)
    return world_corners, (center.x, center.y), rot_handle

def get_shape_points_view(node, ntree):
    w, h = float(node.width), float(node.height)
    center, angle = canvas_element_pose(node, ntree)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    t_int = int(node.armory_type)

    if t_int in {14, 15, 19}:
        segs = 64
        pts = []
        for i in range(segs):
            theta = (2.0 * math.pi * i) / segs
            lx = math.cos(theta) * (w * 0.5)
            ly = math.sin(theta) * (h * 0.5)
            pts.append(transform_point(lx, ly, center, cos_a, sin_a))
        return pts, (t_int == 14)

    elif t_int in {16, 17}:
        local_tri = [
            (-w * 0.5, -h * 0.5),
            ( w * 0.5, -h * 0.5),
            ( 0.0,      h * 0.5),
        ]
        pts = [transform_point(lx, ly, center, cos_a, sin_a) for lx, ly in local_tri]
        return pts, (t_int == 16)

    else:
        local_rect = [
            (-w * 0.5, -h * 0.5),
            ( w * 0.5, -h * 0.5),
            ( w * 0.5,  h * 0.5),
            (-w * 0.5,  h * 0.5),
        ]
        pts = [transform_point(lx, ly, center, cos_a, sin_a) for lx, ly in local_rect]
        filled = t_int not in {0, 3, 13, 19}
        return pts, filled

def point_in_quad(pt, quad):
    x, y = pt
    inside = False
    n = len(quad)
    p1x, p1y = quad[0]
    for i in range(1, n + 1):
        p2x, p2y = quad[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# -------------------------------------------------------------------------
# 4. Pixel-Perfect GPU Visual Canvas Renderer
# -------------------------------------------------------------------------
def snap_px(pt):
    return (round(pt[0]), round(pt[1]))

def snap_px_half(pt):
    return (math.floor(pt[0]) + 0.5, math.floor(pt[1]) + 0.5)

def draw_canvas_element_line(shader, node, points, zoom):
    if node.line_enabled:
        shader.bind()
        draw_thick_closed_loop(shader, points, node.line_thickness * zoom,
                               display_rgba(node.line_color), node.line_position)


def draw_thick_closed_loop(shader, pts_r, thickness_px, color, placement=None):
    n = len(pts_r)
    if n < 3:
        return
    thickness = float(thickness_px) if placement else max(1, float(thickness_px))
    legacy = placement is None
    placement = placement or "CENTER"
    outward = thickness if placement == "OUTSIDE" else thickness * 0.5 if placement == "CENTER" else 0.0
    inward = thickness - outward
    if inward and not legacy:
        inward = min(inward, min(max(p[i] for p in pts_r) - min(p[i] for p in pts_r)
                                 for i in range(2)) * 0.24)
    verts = [Vector((p[0], p[1])) for p in pts_r]

    area2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        area2 += verts[i].x * verts[j].y - verts[j].x * verts[i].y
    ccw = 1.0 if area2 >= 0.0 else -1.0

    outer_pts = []
    inner_pts = []
    for i in range(n):
        p_prev = verts[(i - 1) % n]
        p_curr = verts[i]
        p_next = verts[(i + 1) % n]

        e1 = (p_curr - p_prev)
        e2 = (p_next - p_curr)
        if e1.length < 1e-5 or e2.length < 1e-5:
            outer_pts.append((p_curr.x, p_curr.y))
            inner_pts.append((p_curr.x, p_curr.y))
            continue

        e1.normalize()
        e2.normalize()
        n1 = Vector((e1.y * ccw, -e1.x * ccw))
        n2 = Vector((e2.y * ccw, -e2.x * ccw))
        miter = n1 + n2
        if miter.length < 1e-4:
            miter = n1
        else:
            miter.normalize()
            dot = max(0.25, miter.dot(n1))
            miter *= (1.0 / min(dot, 2.5))

        o_p = p_curr + miter * outward
        i_p = p_curr - miter * inward
        outer_pts.append((o_p.x, o_p.y))
        inner_pts.append((i_p.x, i_p.y))

    tris = []
    for i in range(n):
        j = (i + 1) % n
        o0, o1 = outer_pts[i], outer_pts[j]
        i0, i1 = inner_pts[i], inner_pts[j]
        tris.extend([o0, o1, i1, o0, i1, i0])

    batch = batch_for_shader(shader, 'TRIS', {"pos": tris})
    shader.uniform_float("color", color)
    batch.draw(shader)

def draw_circular_ring_arc(shader, to_reg, node, ntree, ratio, thickness_px, color):
    w, h = float(node.width), float(node.height)
    center, angle = canvas_element_pose(node, ntree)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    ratio = max(0.0, min(1.0, float(ratio)))
    if ratio <= 0.001:
        return

    rx_outer = max(2.0, w * 0.5)
    ry_outer = max(2.0, h * 0.5)
    stroke_w = max(1.0, float(node.strength))
    rx_inner = max(0.5, rx_outer - stroke_w)
    ry_inner = max(0.5, ry_outer - stroke_w)

    segs = max(12, int(round(72 * ratio)))
    start_theta = math.pi * 0.5
    sweep = -2.0 * math.pi * ratio

    tris = []
    for i in range(segs):
        t0 = start_theta + (sweep * i) / segs
        t1 = start_theta + (sweep * (i + 1)) / segs

        c0, s0 = math.cos(t0), math.sin(t0)
        c1, s1 = math.cos(t1), math.sin(t1)

        o0 = to_reg(*transform_point(c0 * rx_outer, s0 * ry_outer, center, cos_a, sin_a))
        o1 = to_reg(*transform_point(c1 * rx_outer, s1 * ry_outer, center, cos_a, sin_a))
        i0 = to_reg(*transform_point(c0 * rx_inner, s0 * ry_inner, center, cos_a, sin_a))
        i1 = to_reg(*transform_point(c1 * rx_inner, s1 * ry_inner, center, cos_a, sin_a))

        tris.extend([o0, o1, i1, o0, i1, i0])

    batch = batch_for_shader(shader, 'TRIS', {"pos": tris})
    shader.uniform_float("color", color)
    batch.draw(shader)

def draw_square_handle(shader, center_r, half_size, fill_col, border_col):
    cx, cy = round(center_r[0]), round(center_r[1])
    hs = round(half_size)
    pts = [
        (cx - hs, cy - hs),
        (cx + hs, cy - hs),
        (cx + hs, cy + hs),
        (cx - hs, cy + hs),
    ]
    batch_f = batch_for_shader(shader, 'TRI_FAN', {"pos": pts})
    shader.uniform_float("color", fill_col)
    batch_f.draw(shader)

    loop_pts = [snap_px_half(p) for p in pts]
    batch_b = batch_for_shader(shader, 'LINE_STRIP', {"pos": loop_pts + [loop_pts[0]]})
    shader.uniform_float("color", border_col)
    gpu.state.line_width_set(1.0)
    batch_b.draw(shader)

def draw_zui_reference_control(shader, node, t_int, corners, zoom):
    """Draw Check, Combo and Slider from the user's Armory visual reference.

    Coordinates are local to the control and support the existing rotated canvas.
    Preview states are Blender-only and never written into Armory JSON.
    """
    origin = Vector(corners[0])
    ex = Vector(corners[1]) - origin
    ey = Vector(corners[3]) - origin
    width, height = ex.length, ey.length
    if width < 1 or height < 1:
        return
    ux, uy = ex / width, ey / height
    def pt(x, y):
        return tuple(origin + ux * x + uy * y)
    def quad(x0, y0, x1, y1):
        return [pt(x0,y0), pt(x1,y0), pt(x1,y1), pt(x0,y1)]
    def fill(points, color):
        shader.bind()
        shader.uniform_float('color', display_rgba(color))
        batch_for_shader(shader, 'TRI_FAN', {'pos': points}).draw(shader)
    def outline(points, color, thick=1.0):
        draw_thick_closed_loop(shader, points, max(1.0, thick*zoom), display_rgba(color))
    def line(points, color):
        shader.bind()
        shader.uniform_float('color', display_rgba(color))
        batch_for_shader(shader, 'LINE_STRIP', {'pos': points}).draw(shader)
    tree = node.id_data
    text_col = canvas_control_text_color(node, t_int)
    bg_col = canvas_element_color(node, t_int)
    hover_col = (tuple(node.color_hover) if node.use_custom_button_states
                 else canvas_theme_color(tree, 'ACCENT_HOVER_COL'))
    if node.disabled and t_int in CANVAS_DISABLED_TYPES:
        hover_col = canvas_grey(hover_col)
    font = get_blf_font_id(node)
    font_px = max(1, round(canvas_node_font_size(node, tree) * zoom))
    blf.size(font, font_px)
    def label(value, x, center_y, align='left', color=None):
        value = str(value)
        tw, th = canvas_text_dimensions(font, font_px, value)
        if align == 'right':
            x -= tw
        elif align == 'center':
            x -= tw * 0.5
        x0, y0 = pt(x, center_y - th * 0.45)
        blf.color(font, *display_rgba(color or text_col))
        blf.position(font, round(x0), round(y0), 0)
        blf.draw(font, value)
    pad = max(5.0, 12.0*zoom)
    if t_int == 9:  # slider: outlined full rectangle + solid proportional fill
        ratio = max(0.0, min(1.0, float(simulation_value(node,"slider",node.preview_slider_value))))
        outline(quad(0,0,width,height), bg_col)
        if ratio > 0:
            fill(quad(0,0,width*ratio,height), hover_col)
        label(canvas_preview_label(node), pad, height*0.5)
        label(f'{ratio:.2f}', width-pad, height*0.5, 'right')
    elif t_int == 8:  # combo: always-on arrow and optional menu below
        outline(quad(0,0,width,height), bg_col)
        options = [x.strip() for x in simulation_options(node) if x.strip()]
        selected = max(0, min(int(simulation_value(node,"index",node.preview_combo_index)), len(options)-1)) if options else 0
        label(options[selected] if options else (canvas_preview_label(node)), pad, height*0.5)
        cx, cy = width - max(13.0*zoom, height*0.35), height*0.5
        arrow = [pt(cx-5*zoom, cy+2*zoom), pt(cx+5*zoom, cy+2*zoom), pt(cx,cy-3*zoom)]
        fill(arrow, bg_col)
        if simulation_value(node,"open",node.preview_combo_open):
            for i, option in enumerate(options[:24]):
                ytop = -i*height
                ybottom = -(i+1)*height
                rect = quad(0,ybottom,width,ytop)
                fill(rect, (0.18,0.19,0.20,0.98) if i != selected else hover_col)
                outline(rect, bg_col)
                label(option, pad, (ytop+ybottom)*0.5)
    elif t_int in {6, 7}:
        side = min(height*0.66, width, 30*zoom)
        y0 = (height-side)*0.5
        dot = t_int == 7 and node.radio_style == 'DOT'
        if t_int == 7:
            selected = simulation_value(node, "radio", node.preview_radio_selected)
            accent = canvas_theme_color(tree, 'ACCENT_SELECT_COL')
            if node.disabled:
                accent = canvas_grey(accent)
            if dot:
                def circle(radius):
                    return [pt(side / 2 + radius * math.cos(i * math.tau / 32),
                               height / 2 + radius * math.sin(i * math.tau / 32)) for i in range(32)]
                outline(circle(side / 2), bg_col)
                if selected:
                    fill(circle(side / 4), bg_col if node.use_element_color else accent)
            else:
                bounds = quad(0, 0, width, height)
                fill(bounds, accent if selected and not node.use_element_color else bg_col)
                if selected and node.use_element_color:
                    outline(bounds, accent, 2.0)
        else:
            outline(quad(0,y0,side,y0+side), bg_col)
        if t_int == 6 and simulation_value(node,"checked",node.preview_checked):
            inset = max(3.0, 5*zoom)
            line([pt(inset,y0+inset), pt(side-inset,y0+side-inset)], text_col)
            line([pt(inset,y0+side-inset), pt(side-inset,y0+inset)], text_col)
        text = canvas_preview_label(node)
        text_width, text_height = canvas_text_dimensions(font, font_px, text)
        padding = canvas_text_padding(node) * zoom
        left = padding if t_int == 7 and not dot else side + padding
        right = max(left, width - padding)
        align = int(node.alignment)
        x = left if align == 0 else max(left, right - text_width)
        if align == 1:
            x = max(left, (left + right - text_width) * 0.5)
        top, bottom = (value * zoom for value in node.text_padding_vertical)
        free = max(0, height - top - bottom - text_height)
        baseline = bottom + free * {'TOP': 1, 'MIDDLE': .5, 'BOTTOM': 0}[node.text_vertical_align]
        label(text, x, baseline + text_height * 0.45)

def canvas_group_badges(tree, view2d, draw_order):
    for node in draw_order:
        if not node_has_children(node) or locked_group_for(node) or not node.is_effectively_visible():
            continue
        corners, _center, _rotate = get_element_corners_view(node, tree)
        x, y = view2d.view_to_region(*corners[2], clip=False)
        yield node, (x + 13, y + 6)


def draw_group_lock_badge(shader, center, locked, parent_locked=False):
    x, y = center
    border = (0.1, 0.12, 0.15, 1.0)
    body = (1.0, 0.65, 0.2, 1.0) if locked else (0.8, 0.84, 0.9, 1.0)
    if parent_locked:
        body = (1.0, 0.16, 0.16, 1.0)
    draw_square_handle(shader, center, 8, (0.23, 0.26, 0.32, 0.95), border)
    shackle = [(x - 3, y), (x - 3, y + 4), (x - 1, y + 6),
               (x + 1, y + 6), (x + 3, y + 4), (x + 3, y if locked else y + 2)]
    shader.uniform_float("color", body)
    batch_for_shader(shader, 'LINE_STRIP', {"pos": shackle}).draw(shader)
    draw_square_handle(shader, (x, y - 3), 3.5, body, border)


_canvas_image_shader = None


def canvas_image_shader():
    global _canvas_image_shader
    if _canvas_image_shader is None:
        interface = gpu.types.GPUStageInterfaceInfo("armory_canvas_image")
        interface.smooth('VEC2', "uv")
        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant('MAT4', "transform")
        info.push_constant('VEC4', "tint")
        info.sampler(0, 'FLOAT_2D', "image")
        info.vertex_in(0, 'VEC2', "pos")
        info.vertex_in(1, 'VEC2', "texCoord")
        info.vertex_out(interface)
        info.fragment_out(0, 'VEC4', "fragColor")
        info.vertex_source("void main() { uv = texCoord; gl_Position = transform * vec4(pos, 0.0, 1.0); }")
        info.fragment_source("void main() { fragColor = texture(image, uv) * tint; }")
        _canvas_image_shader = gpu.shader.create_from_info(info)
    return _canvas_image_shader


def draw_armory_canvas_callback():
    context = bpy.context
    space = context.space_data
    if not space or space.type != 'NODE_EDITOR' or space.tree_type != 'ArmoryCanvasTreeType':
        return

    ntree = space.edit_tree
    if not ntree or ntree.bl_idname != 'ArmoryCanvasTreeType' or not ntree.show_canvas_overlay:
        return

    region = context.region
    if not region or region.type != 'WINDOW':
        return

    started = perf_counter() if ntree.show_viewport_fps else None
    try:
        draw_armory_canvas(ntree, region)
        if started is not None:
            draw_canvas_debug_fps(ntree, region, started)
    finally:
        gpu.state.line_width_set(1.0)
        gpu.state.point_size_set(1.0)
        gpu.state.blend_set('NONE')


def draw_armory_canvas(ntree, region):
    global _canvas_draw_cache
    previous = _canvas_draw_cache
    theme = canvas_current_theme(ntree)
    _canvas_draw_cache = canvas_geometry_cache(ntree)
    _canvas_draw_cache['theme'] = theme
    try:
        draw_armory_canvas_contents(ntree, region)
    finally:
        canvas_restore_scissor()
        _canvas_draw_cache = previous


def draw_canvas_debug_fps(tree, region, started):
    draw_ms = (perf_counter() - started) * 1000
    key = (region.as_pointer(), tree.as_pointer())
    previous, smooth_interval = _canvas_frame_samples.get(key, (started, None))
    interval = started - previous
    if 0.0 < interval < 0.75:
        smooth_interval = interval if smooth_interval is None else 0.8 * smooth_interval + 0.2 * interval
    else:
        smooth_interval = None
    if len(_canvas_frame_samples) >= 32 and key not in _canvas_frame_samples:
        _canvas_frame_samples.pop(next(iter(_canvas_frame_samples)))
    _canvas_frame_samples[key] = started, smooth_interval
    fps = f"{1 / smooth_interval:.0f}" if smooth_interval else "—"
    label = f"Viewport {fps} FPS   |   Canvas CPU {draw_ms:.1f} ms"
    gpu.state.blend_set('ALPHA')
    blf.size(0, 14)
    blf.enable(0, blf.SHADOW)
    try:
        blf.shadow(0, 3, 0, 0, 0, 1)
        blf.shadow_offset(0, 1, -1)
        blf.position(0, 14, region.height - 26, 0)
        blf.color(0, 1, 1, 1, 1)
        blf.draw(0, label)
    finally:
        blf.disable(0, blf.SHADOW)


def canvas_hierarchy_cache(tree):
    key = tree.session_uid
    cached = _canvas_hierarchies.get(key)
    if cached is not None:
        return cached
    parents = {}
    for link in tree.links:
        if link.from_socket.name == "Parent":
            parents.setdefault(link.from_node, link.to_node)
        elif link.to_socket.name == "Parent":
            parents.setdefault(link.to_node, link.from_node)
    nodes = get_ordered_canvas_nodes(tree)
    children = {}
    for node in nodes:
        parent = parents.get(node)
        if isinstance(parent, ArmoryCanvasNodeBase):
            children.setdefault(parent, []).append(node)
        else:
            parents.pop(node, None)
    for siblings in children.values():
        siblings.sort(key=lambda node: (node.z_order, node.name))
    hierarchy = build_hierarchical_outliner_order(tree, nodes, parents)
    cached = {
        'tree': tree, 'nodes': nodes, 'parents': parents, 'children': children,
        'hierarchy': hierarchy,
        'ranks': {node: rank for rank, (node, _, _) in enumerate(hierarchy)},
        'depths': {node: depth for node, depth, _ in hierarchy},
    }
    if len(_canvas_hierarchies) >= 16:
        oldest = next(iter(_canvas_hierarchies))
        _canvas_hierarchies.pop(oldest)
        _canvas_geometry.pop(oldest, None)
    _canvas_hierarchies[key] = cached
    return cached


def canvas_geometry_cache(tree):
    key = tree.session_uid
    cache = _canvas_geometry.get(key)
    if cache is None:
        hierarchy = canvas_hierarchy_cache(tree)
        cache = {
            **hierarchy, 'topology': hierarchy, 'corners': {}, 'groups': {}, 'scroll': {},
            'has_scroll': any(node.mask_children or canvas_scroll_enabled(node)
                              for node in hierarchy['nodes']),
            'original_scissor': None,
        }
        _canvas_geometry[key] = cache
    return cache


def canvas_grey(color):
    r, g, b, alpha = color
    grey = (r * .2126 + g * .7152 + b * .0722) * .6 + .12
    return grey, grey, grey, alpha


@lru_cache(maxsize=2048)
def canvas_text_dimensions(font, size, text):
    # Callers set this BLF font to size before using its metrics.
    return blf.dimensions(font, text)


@lru_cache(maxsize=512)
def canvas_wrapped_text(font, size, text, width):
    return canvas_wrap_lines(text, width, lambda value: canvas_text_dimensions(font, size, value)[0])


def canvas_wrap_lines(text, width, measure):
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        words = paragraph.split()
        for word in words:
            candidate = f"{line} {word}" if line else word
            if line and measure(candidate) > width:
                lines.append(line)
                line = ""
                candidate = word
            if measure(candidate) <= width:
                line = candidate
                continue
            chunk = ""
            for character in word:
                if chunk and measure(chunk + character) > width:
                    lines.append(chunk)
                    chunk = ""
                chunk += character
            line = chunk
        lines.append(line)
    return lines


def canvas_slice_axis(source, destination, leading, trailing):
    scale = min(1, source / max(1, leading + trailing))
    leading, trailing = leading * scale, trailing * scale
    shrink = min(1, destination / max(1, leading + trailing))
    return ((0, leading / source, 1 - trailing / source, 1),
            (0, leading * shrink / destination, 1 - trailing * shrink / destination, 1))


def canvas_slice_geometry(corners, image_size, destination_size, borders):
    left, right, top, bottom = borders
    ux, dx = canvas_slice_axis(image_size[0], destination_size[0], left, right)
    uy, dy = canvas_slice_axis(image_size[1], destination_size[1], bottom, top)
    start = Vector(corners[0])
    horizontal, vertical = Vector(corners[1]) - start, Vector(corners[3]) - start
    points, uv = [], []
    for row in range(3):
        for column in range(3):
            if dx[column] >= dx[column + 1] or dy[row] >= dy[row + 1] or ux[column] >= ux[column + 1] or uy[row] >= uy[row + 1]:
                continue
            for x, y in ((column, row), (column + 1, row), (column + 1, row + 1),
                         (column, row), (column + 1, row + 1), (column, row + 1)):
                points.append(tuple(start + horizontal * dx[x] + vertical * dy[y]))
                uv.append((ux[x], uy[y]))
    return points, uv


def canvas_rounded_points(corners, radius):
    r = min(radius, (Vector(corners[1]) - Vector(corners[0])).length / 2,
            (Vector(corners[3]) - Vector(corners[0])).length / 2)
    if r <= 0:
        return corners
    result = []
    for i, corner in enumerate(map(Vector, corners)):
        along = Vector(corners[(i + 1) % 4]) - corner
        across = Vector(corners[i - 1]) - corner
        if along.length == 0 or across.length == 0:
            return corners
        along.normalize()
        across.normalize()
        center = corner + (along + across) * r
        for step in range(9):
            angle = math.pi * step / 16
            result.append(tuple(center - r * (along * math.cos(angle) + across * math.sin(angle))))
    return result


def canvas_scroll_gutters(node, width, height):
    if not canvas_scroll_enabled(node):
        return 0, 0
    return tuple(min(size - 1, requested) if enabled else 0
                 for size, requested, enabled in zip((width, height), node.scroll_bar_size, reversed(node.scroll_axes)))


def canvas_scroll_viewport_size(node, width, height):
    right, bottom = canvas_scroll_gutters(node, width, height)
    return width - right, height - bottom


def canvas_scroll_range(node):
    cache = _canvas_draw_cache
    if cache is not None and node in cache['scroll']:
        return cache['scroll'][node]
    width, height = canvas_scroll_viewport_size(node, node.width, node.height)
    far = [max(width, node.scroll_content[0]), max(height, node.scroll_content[1])]
    lookup = cache['children'] if cache is not None else None

    def visit(parent, x, y, visited):
        children = lookup.get(parent, ()) if lookup is not None else get_direct_children(node.id_data, parent)
        for child in children:
            if not child.visible or child in visited:
                continue
            box = layout_child_box(node.id_data, child)
            if box is None:
                w, h = child.width, child.height
                anchor = int(child.anchor)
                cx = child.pos_x + round((parent.width - w) * (anchor % 3) / 2)
                cy = child.pos_y + round((parent.height - h) * (anchor // 3) / 2)
            else:
                cx, cy, w, h = box
            cx += x
            cy += y
            scale_x, scale_y = child.offset_scale_xy if (box is not None and parent.layout_use_child_scale
                and parent.radio_layout_mode in {'HORIZONTAL', 'VERTICAL'} and not child.offset_visual_only) else (1, 1)
            far[0] = max(far[0], cx + w * (1 + scale_x) / 2)
            far[1] = max(far[1], cy + h * (1 + scale_y) / 2)
            if not child.mask_children and not canvas_scroll_enabled(child):
                visit(child, cx, cy, visited | {child})

    visit(node, 0, 0, {node})
    result = tuple(max(0, far[i] - size) if node.scroll_axes[i] else 0
                   for i, size in enumerate((width, height)))
    if cache is not None:
        cache['scroll'][node] = result
    return result


def canvas_scroll_enabled(node):
    return int(node.armory_type) in LAYOUT_CONTAINER_TYPES and any(node.scroll_axes)


def canvas_scroll_offset(node):
    offsets = simulation_value(node, "scroll", node.preview_scroll)
    if not canvas_scroll_enabled(node) or not any(offsets):
        return 0, 0
    return tuple(round(min(limit, value)) for limit, value in zip(canvas_scroll_range(node), offsets))


def canvas_scroll_clip(tree, node):
    if _canvas_draw_cache is not None and not _canvas_draw_cache['has_scroll']:
        return None
    parent = get_parent_node(node)
    clip = None
    seen = {node}
    while parent is not None and parent not in seen:
        if parent.mask_children or canvas_scroll_enabled(parent):
            bounds = canvas_scroll_rect(tree, parent, content=True)
            clip = bounds if clip is None else canvas_intersect_bounds(clip, bounds)
        seen.add(parent)
        parent = get_parent_node(parent)
    return clip


def canvas_has_offset(node):
    return bool(any(node.offset_position) or node.offset_rotation or tuple(node.offset_scale_xy) != (1, 1))


def canvas_interaction_corners(tree, node):
    corners, center, _ = get_element_corners_view(node, tree)
    if node.offset_visual_only or not canvas_has_offset(node):
        return corners
    center = Vector(center)
    dx, dy = node.offset_position
    scale_x, scale_y = node.offset_scale_xy
    cos_a = math.cos(-node.offset_rotation)
    sin_a = math.sin(-node.offset_rotation)
    origin = center + Vector((dx, -dy))
    return [transform_point((x - center.x) * scale_x, (y - center.y) * scale_y,
                            origin, cos_a, sin_a) for x, y in corners]


def canvas_visible_bounds(tree, node, *, interaction=False):
    corners = canvas_interaction_corners(tree, node) if interaction else get_element_corners_view(node, tree)[0]
    bounds = canvas_bounds(corners)
    clip = canvas_scroll_clip(tree, node)
    if clip is not None:
        bounds = canvas_intersect_bounds(bounds, clip)
    return bounds if all(lo <= hi for lo, hi in bounds) else None


def canvas_point_in_visible_node(tree, node, point, *, interaction=False):
    clip = canvas_scroll_clip(tree, node)
    if clip is not None and not all(lo <= point[axis] <= hi for axis, (lo, hi) in enumerate(clip)):
        return False
    corners = canvas_interaction_corners(tree, node) if interaction else get_element_corners_view(node, tree)[0]
    return point_in_quad(point, corners)


def canvas_restore_scissor():
    scissor = _canvas_draw_cache['original_scissor']
    if scissor is None:
        return
    gpu.state.scissor_set(*scissor)
    gpu.state.scissor_test_set(scissor[2] > 0 and scissor[3] > 0)
    _canvas_draw_cache['original_scissor'] = None


def canvas_scroll_rect(tree, node, *, content=False):
    x, y = node.get_world_pos(tree)
    top = canvas_view_top(tree) - y
    width, height = node.width, node.height
    if content:
        width, height = canvas_scroll_viewport_size(node, width, height)
    return (x, x + width), (top - height, top)


def canvas_scroll_bars(tree, node):
    if not canvas_scroll_enabled(node):
        return
    rect = canvas_scroll_rect(tree, node)
    limits = canvas_scroll_range(node)
    offsets = canvas_scroll_offset(node)
    gutters = canvas_scroll_gutters(node, node.width, node.height)
    for axis, limit in enumerate(limits):
        if not node.scroll_axes[axis]:
            continue
        low, high = rect[axis]
        thickness = gutters[1 - axis]
        track = max(1, high - low - gutters[axis])
        handle = min(track, max(thickness, track * track / (track + limit)))
        travel = track - handle
        progress = offsets[axis] / limit if limit else 0
        if axis == 0:
            cross = (rect[1][0], rect[1][0] + thickness)
            bound = (low + travel * progress, low + travel * progress + handle)
            yield axis, ((low, low + track), cross), (bound, cross), travel, limit
        else:
            cross = (rect[0][1] - thickness, rect[0][1])
            bound = (high - travel * progress - handle, high - travel * progress)
            yield axis, (cross, (high - track, high)), (cross, bound), travel, limit


class NODE_OT_armory_canvas_preview_scroll(Operator):
    bl_idname = "node.armory_canvas_preview_scroll"
    bl_label = "Scroll Canvas Preview"

    def invoke(self, context, event):
        tree = canvas_shortcut_tree(context)
        if tree is None or not tree.show_canvas_overlay or context.region.type != 'WINDOW':
            return {'PASS_THROUGH'}
        view = context.region.view2d
        point = view.region_to_view(event.mouse_region_x, event.mouse_region_y)
        for node in reversed(canvas_paint_order(tree)):
            if not canvas_scroll_enabled(node) or not node.is_effectively_visible():
                continue
            bounds = canvas_scroll_rect(tree, node)
            clip = canvas_scroll_clip(tree, node)
            if clip is not None:
                bounds = canvas_intersect_bounds(bounds, clip)
            if not all(lo <= point[axis] <= hi for axis, (lo, hi) in enumerate(bounds)):
                continue
            limits = canvas_scroll_range(node)
            axis = 0 if event.shift or not limits[1] else 1
            if not limits[axis]:
                continue
            change = -40 if event.type == 'WHEELUPMOUSE' else 40
            start = canvas_scroll_offset(node)[axis]
            target = max(0, min(limits[axis], start + change))
            if start != target:
                node.preview_scroll[axis] = target
                context.area.tag_redraw()
                return {'FINISHED'}
        return {'PASS_THROUGH'}


def canvas_draw_special_control(shader, node, corners, zoom, variant):
    origin = Vector(corners[0])
    horizontal, vertical = Vector(corners[1]) - origin, Vector(corners[3]) - origin
    width, height = horizontal.length, vertical.length
    if not width or not height:
        return
    def rect(x, y, w, h):
        return [tuple(origin + horizontal * (px / width) + vertical * (py / height))
                for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))]
    def fill(points, color):
        shader.bind()
        shader.uniform_float('color', display_rgba(canvas_grey(color) if node.disabled else color))
        batch_for_shader(shader, 'TRI_FAN', {'pos': points}).draw(shader)
    session = _canvas_simulations.get(node.id_data.as_pointer()) or {}
    def end_color(side, default):
        if not node.disabled and session.get('hover') == node.name and session.get('hover_end') == side:
            if session.get('pressed') == node.name and session.get('pressed_end') == side:
                return node.color_press if node.use_custom_button_states else canvas_theme_color(node.id_data, 'BUTTON_PRESSED_COL')
            return node.color_hover if node.use_custom_button_states else canvas_theme_color(node.id_data, 'BUTTON_HOVER_COL')
        return node.end_color if node.use_end_color else default

    def label(text, left, right, aligned=False):
        if right <= left:
            return
        font = get_blf_font_id(node)
        blf.size(font, max(1, round(min(height * .7, canvas_node_font_size(node, node.id_data) * zoom))))
        tw, th = blf.dimensions(font, text)
        x = (left + right - tw) / 2
        y = (height - th) / 2
        if aligned:
            pad = canvas_text_padding(node) * zoom
            if node.alignment == '0':
                x = left + pad
            elif node.alignment == '2':
                x = right - pad - tw
            if node.text_vertical_align == 'TOP':
                y = height - th
            elif node.text_vertical_align == 'BOTTOM':
                y = 0
        point = origin + horizontal * (x / width) + vertical * (y / height)
        blf.position(font, point.x, point.y, 0)
        blf.color(font, *display_rgba(canvas_control_text_color(node, int(node.armory_type))))
        blf.rotation(font, math.atan2(horizontal.y, horizontal.x))
        blf.enable(font, blf.ROTATION)
        try:
            blf.draw(font, text)
        finally:
            blf.disable(font, blf.ROTATION)
    if variant == 'STEPPER':
        edge = min(height, width / 3)
        color = canvas_element_color(node, 2)
        radius = node.corner_radius * zoom
        fill(canvas_rounded_points(rect(0, 0, width, height), radius), color)
        shade = canvas_theme_color(node.id_data, 'ACCENT_HOVER_COL')
        fill(canvas_rounded_points(rect(0, 0, edge, height), radius), end_color(-1, shade))
        fill(canvas_rounded_points(rect(width - edge, 0, edge, height), radius), end_color(1, shade))
        label("−", 0, edge)
        label(f"{canvas_step_value(node):.7g}", edge, width - edge, True)
        label("+", width - edge, width)
    else:
        active = simulation_value(node, 'switch', node.get('_preview_switch_value', node.switch_initial))
        track = min(height * .66, width / 2)
        offset = (height - track) / 2
        color = canvas_theme_color(node.id_data, 'HIGHLIGHT_COL' if active else 'ACCENT_COL')
        fill(canvas_rounded_points(rect(0, offset, track * 2, track), track / 2), color)
        inset = track * .12
        thumb = track - 2 * inset
        fill(canvas_rounded_points(rect(inset + (track if active else 0), offset + inset, thumb, thumb),
                                   thumb / 2), canvas_control_text_color(node, 6))
        if node.text_content:
            label(node.text_content, track * 2, width, True)


def draw_canvas_grid(shader, tree, region, width, height, top):
    view = region.view2d
    step = tree.grid_size
    to_region = view.view_to_region
    if abs(to_region(step, 0, clip=False)[0] - to_region(0, 0, clip=False)[0]) < 4:
        return

    left, bottom = view.region_to_view(0, 0)
    right, upper = view.region_to_view(region.width, region.height)
    key = (left, bottom, right, upper, region.width, region.height,
           width, height, top, step, tree.extend_grid)
    region_key = region.as_pointer()
    cached_key, batch = _canvas_grid_batches.get(region_key, (None, None))
    if key != cached_key:
        first_x = math.floor(left / step)
        last_x = math.ceil(right / step)
        first_y = math.floor((top - upper) / step)
        last_y = math.ceil((top - bottom) / step)
        if not tree.extend_grid:
            first_x, last_x = max(1, first_x), min(math.ceil(width / step) - 1, last_x)
            first_y, last_y = max(1, first_y), min(math.ceil(height / step) - 1, last_y)
            left, right = 0, width
            bottom, upper = top - height, top

        lines = []
        for index in range(first_x, last_x + 1):
            x = index * step
            lines.extend(snap_px_half(to_region(x, y, clip=False)) for y in (bottom, upper))
        for index in range(first_y, last_y + 1):
            y = top - index * step
            lines.extend(snap_px_half(to_region(x, y, clip=False)) for x in (left, right))
        batch = batch_for_shader(shader, 'LINES', {"pos": lines}) if lines else None
        if len(_canvas_grid_batches) >= 16 and region_key not in _canvas_grid_batches:
            _canvas_grid_batches.pop(next(iter(_canvas_grid_batches)))
        _canvas_grid_batches[region_key] = key, batch
    if batch is not None:
        gpu.state.line_width_set(1.0)
        shader.uniform_float("color", display_rgba(tree.grid_color))
        batch.draw(shader)


def draw_armory_canvas_contents(ntree, region):
    v2d = region.view2d
    cw, height = canvas_preview_size(ntree)
    ch = canvas_view_top(ntree)
    bottom = ch - height

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')

    def to_reg(vx, vy):
        return v2d.view_to_region(vx, vy, clip=False)

    art_pts = [snap_px(to_reg(0, bottom)), snap_px(to_reg(cw, bottom)), snap_px(to_reg(cw, ch)), snap_px(to_reg(0, ch))]
    batch_bg = batch_for_shader(shader, 'TRI_FAN', {"pos": art_pts})
    shader.bind()
    shader.uniform_float("color", display_rgba(ntree.canvas_bg_color))
    batch_bg.draw(shader)

    if ntree.show_grid and ntree.grid_size >= 2:
        draw_canvas_grid(shader, ntree, region, cw, height, ch)

    art_loop = [snap_px_half(p) for p in art_pts]
    art_loop.append(art_loop[0])
    batch_border = batch_for_shader(shader, 'LINE_STRIP', {"pos": art_loop})
    shader.uniform_float("color", display_rgba(ntree.canvas_border_color))
    gpu.state.line_width_set(1.0)
    batch_border.draw(shader)

    tl_reg = snap_px(to_reg(0, ch))
    font_id = 0
    blf.position(font_id, tl_reg[0], tl_reg[1] + (24 if ntree.show_element_names else 8), 0)
    blf.size(font_id, 13)
    b_col = ntree.canvas_border_color
    blf.color(font_id, *display_rgba(b_col))
    status_suffix = " *" if ntree.is_dirty else ""
    vis_warn = " [CANVAS HIDDEN]" if not ntree.canvas_visible else ""
    grid_info = f" | Grid: {ntree.grid_size}px" if ntree.show_grid else ""
    blf.draw(font_id, f"{ntree.canvas_name} [{int(cw)}x{int(height)}] - {ntree.theme_name}{grid_info}{vis_warn}{status_suffix}")

    active_node = ntree.nodes.active
    show_gizmos = ntree.show_gizmos and ntree.as_pointer() not in _canvas_simulations

    zoom_scale = max(0.15, min(4.0, abs(to_reg(100, 0)[0] - to_reg(0, 0)[0]) / 100.0))

    draw_order = canvas_paint_order(ntree)
    scissor = None
    region_bounds = ((0, region.width), (0, region.height))

    def apply_clip(clip):
        nonlocal scissor
        if scissor == clip:
            return clip is None or all(lo < hi for lo, hi in clip)
        scissor = clip
        if clip is None:
            canvas_restore_scissor()
            return True
        (left, right), (bottom, top) = clip
        if left >= right or bottom >= top:
            return False
        if _canvas_draw_cache['original_scissor'] is None:
            _canvas_draw_cache['original_scissor'] = gpu.state.scissor_get()
        origin_x, origin_y, _, _ = gpu.state.viewport_get()
        gpu.state.scissor_set(origin_x + left, origin_y + bottom, right - left, top - bottom)
        gpu.state.scissor_test_set(True)
        return True

    def clip_node(node):
        clip = canvas_scroll_clip(ntree, node)
        if clip is None:
            return apply_clip(None)
        screen = canvas_bounds([to_reg(*point) for point in canvas_box_points(clip)])
        if any(low >= high for low, high in clip):
            return False
        return apply_clip(canvas_intersect_bounds(screen, region_bounds))

    def draw_element(node):
        t_int = int(node.armory_type)
        corners_v, center_v, rot_v = get_element_corners_view(node, ntree)
        corners_r = [snap_px(to_reg(x, y)) for (x, y) in corners_v]
        # Clipped text and controls with no overflowing labels have bounded visuals.
        bounded = t_int in {1, 3, 12, 13, 14, 15, 16, 17, 18, 19}
        if t_int in {0, 20}:
            _, angle = canvas_element_pose(node, ntree)
            bounded = abs(math.remainder(angle, 2.0 * math.pi)) < 1e-5
        elif t_int == 2:
            bounded = node.control_variant != 'STEPPER' and not node.text_content
        elif t_int in {10, 11}:
            bounded = not canvas_preview_label(node)
        if bounded and not canvas_has_offset(node):
            margin = 4 + (node.strength + (node.line_thickness if node.line_enabled else 0)) * zoom_scale
            if node.shadow_enabled:
                margin += max(map(abs, node.shadow_offset)) * zoom_scale
            if any(max(p[i] for p in corners_r) < -margin or min(p[i] for p in corners_r) > limit + margin
                   for i, limit in enumerate((region.width, region.height))):
                return

        # BLF and image draws can change render state between elements.
        gpu.state.blend_set('ALPHA')
        shader.bind()
        if t_int in {14, 15, 16, 17, 19}:
            shape_v, is_filled = get_shape_points_view(node, ntree)
            shape_r = [snap_px(to_reg(x, y)) for (x, y) in shape_v]
        else:
            shape_v, shape_r = corners_v, corners_r
            is_filled = t_int not in {0, 3, 13}
        if t_int in CANVAS_RADIUS_TYPES and node.corner_radius:
            shape_r = canvas_rounded_points(corners_r, node.corner_radius * zoom_scale)
        if t_int in CANVAS_STYLE_TYPES and node.shadow_enabled:
            dx, dy = node.shadow_offset
            shadow = [(x + dx * zoom_scale, y - dy * zoom_scale) for x, y in shape_r]
            if is_filled or t_int == 2:
                shader.uniform_float("color", display_rgba(node.shadow_color))
                batch_for_shader(shader, 'TRI_FAN', {"pos": shadow}).draw(shader)
            else:
                draw_thick_closed_loop(shader, shadow, max(1, node.strength * zoom_scale),
                                       display_rgba(node.shadow_color))

        if t_int == 3:
            stroke_pts = [snap_px_half(to_reg(x, y)) for (x, y) in shape_v]
            stroke_loop = stroke_pts + [stroke_pts[0]]
            batch_stroke = batch_for_shader(shader, 'LINE_STRIP', {"pos": stroke_loop})
            shader.uniform_float("color", (0.25, 0.85, 0.95, 0.55))
            gpu.state.line_width_set(1.0)
            batch_stroke.draw(shader)
            draw_canvas_element_line(shader, node, shape_r, zoom_scale)
            return

        r, g, b, a = canvas_element_color(node, t_int)
        if t_int == 11 and node.preview_key_hover:
            r, g, b, a = (node.color_hover if node.use_custom_button_states
                          else canvas_theme_color(ntree, 'ACCENT_HOVER_COL'))

        variant = node.control_variant
        if (t_int == 2 and variant == 'STEPPER') or (t_int == 6 and variant == 'SWITCH'):
            canvas_draw_special_control(shader, node, corners_r, zoom_scale, variant)
            draw_canvas_element_line(shader, node, shape_r, zoom_scale)
            return
        if t_int in {6, 7, 8, 9}:
            draw_zui_reference_control(shader, node, t_int, corners_r, zoom_scale)
            draw_canvas_element_line(shader, node, shape_r, zoom_scale)
            return

        prog_total = max(1, int(node.progress_total))
        prog_at = max(0, min(prog_total, int(node.progress_at)))
        prog_ratio = max(0.0, min(1.0, prog_at / float(prog_total)))
        pr, pg, pb, pa = node.color_progress if node.use_progress_color else canvas_theme_color(ntree, 'ACCENT_COL')

        if t_int == 19:
            # Inner disc plus external annular progress, matching the requested CProgress layout.
            # Keep the disc inside the ring and preserve the node's aspect ratio.
            cx, cy = center_v
            inner_rx = max(1.0, node.width * 0.5 - max(1.0, node.strength) - 3.0)
            inner_ry = max(1.0, node.height * 0.5 - max(1.0, node.strength) - 3.0)
            inner_pts = [to_reg(cx + inner_rx * math.cos(2 * math.pi * k / 64),
                                cy + inner_ry * math.sin(2 * math.pi * k / 64)) for k in range(64)]
            shader.bind()
            shader.uniform_float("color", display_rgba((r, g, b, a)))
            batch_for_shader(shader, 'TRI_FAN', {"pos": inner_pts}).draw(shader)
            stroke_px = max(1.0, float(node.strength) * zoom_scale)
            draw_circular_ring_arc(shader, to_reg, node, ntree, 1.0, stroke_px, display_rgba((r, g, b, a)))
            if prog_ratio > 0.0:
                draw_circular_ring_arc(shader, to_reg, node, ntree, prog_ratio, stroke_px, display_rgba((pr, pg, pb, pa)))
            draw_canvas_element_line(shader, node, shape_r, zoom_scale)
            return

        drawn_image = False
        img_db = get_canvas_image(node) if t_int == 1 else node.button_image if t_int == 2 else None
        if img_db is not None:
            try:
                gpu_tex = gpu.texture.from_image(img_db)
                if gpu_tex is not None:
                    img_corners_r = corners_r
                    if t_int == 1 and node.keep_aspect_ratio and not node.slice_enabled:
                        img_w, img_h = img_db.size[0], img_db.size[1]
                        elem_w, elem_h = float(node.width), float(node.height)
                        if img_w > 0 and img_h > 0 and elem_w > 0 and elem_h > 0:
                            img_aspect = img_w / float(img_h)
                            elem_aspect = elem_w / elem_h
                            if img_aspect > elem_aspect:
                                fit_w = elem_w
                                fit_h = elem_w / img_aspect
                            else:
                                fit_h = elem_h
                                fit_w = elem_h * img_aspect

                            align_mode = int(node.alignment)
                            if align_mode == 1:
                                off_lx = (elem_w - fit_w) * 0.5
                            elif align_mode == 2:
                                off_lx = elem_w - fit_w
                            else:
                                off_lx = 0.0
                            off_ly = (elem_h - fit_h) * 0.5

                            center, angle = canvas_element_pose(node, ntree)
                            cos_a, sin_a = math.cos(angle), math.sin(angle)

                            lx0 = -elem_w * 0.5 + off_lx
                            ly0 = -elem_h * 0.5 + off_ly
                            fit_local = [
                                (lx0,         ly0),
                                (lx0 + fit_w, ly0),
                                (lx0 + fit_w, ly0 + fit_h),
                                (lx0,         ly0 + fit_h),
                            ]
                            fit_world = [transform_point(lx, ly, center, cos_a, sin_a) for lx, ly in fit_local]
                            img_corners_r = [snap_px(to_reg(x, y)) for (x, y) in fit_world]

                    is_premul = getattr(img_db, "alpha_mode", 'STRAIGHT') == 'PREMUL'
                    gpu.state.blend_set('ALPHA_PREMULT' if is_premul else 'ALPHA')
                    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                    if t_int == 2 or node.use_element_color:
                        tint = (r * a, g * a, b * a, a) if is_premul else (r, g, b, a)
                    else:
                        tint = (1.0, 1.0, 1.0, 1.0)
                    img_shader = canvas_image_shader()
                    primitive = 'TRI_FAN'
                    if node.slice_enabled and all(img_db.size):
                        img_corners_r, uvs = canvas_slice_geometry(
                            corners_r, img_db.size, (node.width, node.height), node.slice_borders)
                        primitive = 'TRIS'
                    batch_img = batch_for_shader(
                        img_shader, primitive, {"pos": img_corners_r, "texCoord": uvs})
                    img_shader.bind()
                    transform = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
                    img_shader.uniform_float("transform", transform)
                    img_shader.uniform_sampler("image", gpu_tex)
                    img_shader.uniform_float("tint", tint)
                    batch_img.draw(img_shader)
                    gpu.state.blend_set('ALPHA')
                    drawn_image = True
                    shader.bind()
            except Exception:
                gpu.state.blend_set('ALPHA')
                drawn_image = False

        if t_int == 20:
            background = (r, g, b, a) if node.use_element_color else canvas_theme_color(ntree, 'SEPARATOR_COL')
            shader.bind()
            shader.uniform_float("color", display_rgba(background))
            batch_for_shader(shader, 'TRI_FAN', {"pos": corners_r}).draw(shader)
        elif t_int in INPUT_OUTLINE_TYPES and not drawn_image:
            shader.bind()
            draw_thick_closed_loop(shader, corners_r, max(1.0, zoom_scale), display_rgba((r, g, b, a)))
        if is_filled and not drawn_image and t_int not in {0, 8, 9} | INPUT_OUTLINE_TYPES:
            batch_fill = batch_for_shader(shader, 'TRI_FAN', {"pos": shape_r})
            shader.bind()
            shader.uniform_float("color", display_rgba((r, g, b, a)))
            batch_fill.draw(shader)

        if not drawn_image and t_int not in {0, 8, 9, 18} | INPUT_OUTLINE_TYPES:
            if t_int in {13, 15, 17}:
                stroke_px = max(1.0, float(node.strength) * zoom_scale)
                draw_thick_closed_loop(shader, shape_r, stroke_px, display_rgba((r, g, b, a)))
            elif not is_filled:
                stroke_pts = [snap_px_half(to_reg(x, y)) for (x, y) in shape_v]
                stroke_loop = stroke_pts + [stroke_pts[0]]
                batch_stroke = batch_for_shader(shader, 'LINE_STRIP', {"pos": stroke_loop})
                shader.uniform_float("color", (0.15, 0.15, 0.15, 0.65))
                gpu.state.line_width_set(1.0)
                batch_stroke.draw(shader)

        if t_int == 18:
            draw_thick_closed_loop(shader, corners_r, max(1.0, float(node.strength) * zoom_scale), display_rgba((r, g, b, a)))
        if t_int == 18 and prog_ratio > 0.0:
            p_pts = [
                corners_r[0],
                (corners_r[0][0] + (corners_r[1][0] - corners_r[0][0]) * prog_ratio,
                 corners_r[0][1] + (corners_r[1][1] - corners_r[0][1]) * prog_ratio),
                (corners_r[3][0] + (corners_r[2][0] - corners_r[3][0]) * prog_ratio,
                 corners_r[3][1] + (corners_r[2][1] - corners_r[3][1]) * prog_ratio),
                corners_r[3]
            ]
            batch_prog = batch_for_shader(shader, 'TRI_FAN', {"pos": p_pts})
            shader.uniform_float("color", display_rgba((pr, pg, pb, pa)))
            batch_prog.draw(shader)

        if t_int == 9:
            # Thin horizontal slider track and proportional filled section.
            ratio = max(0.0, min(1.0, float(simulation_value(node,"slider",node.preview_slider_value))))
            left_mid = ((corners_r[0][0]+corners_r[3][0])*0.5,
                        (corners_r[0][1]+corners_r[3][1])*0.5)
            right_mid = ((corners_r[1][0]+corners_r[2][0])*0.5,
                         (corners_r[1][1]+corners_r[2][1])*0.5)
            direction = Vector((right_mid[0]-left_mid[0], right_mid[1]-left_mid[1]))
            if direction.length > 0.001:
                normal = Vector((-direction.y, direction.x)).normalized() * max(2.0, 3.0*zoom_scale)
                a0, a1 = Vector(left_mid), Vector(right_mid)
                def track_quad(p0, p1):
                    return [tuple(p0+normal), tuple(p1+normal), tuple(p1-normal), tuple(p0-normal)]
                shader.bind()
                shader.uniform_float("color", display_rgba((r, g, b, a)))
                batch_for_shader(shader, 'TRI_FAN', {"pos": track_quad(a0,a1)}).draw(shader)
                if ratio > 0:
                    shader.uniform_float("color", display_rgba((pr, pg, pb, pa)))
                    batch_for_shader(shader, 'TRI_FAN', {"pos": track_quad(a0,a0+(a1-a0)*ratio)}).draw(shader)

        if t_int == 8:
            # Always-on dropdown button, outlined rather than filled.
            right = corners_r[1]
            bottom = corners_r[2]
            top_left = corners_r[0]
            button_ratio = min(0.35, max(0.12, 30.0 / max(1.0, float(node.width))))
            left_top = (right[0] + (top_left[0]-right[0])*button_ratio,
                        right[1] + (top_left[1]-right[1])*button_ratio)
            left_bottom = (bottom[0] + (corners_r[3][0]-bottom[0])*button_ratio,
                           bottom[1] + (corners_r[3][1]-bottom[1])*button_ratio)
            shader.bind()
            shader.uniform_float("color", display_rgba((r, g, b, a)))
            batch_for_shader(shader, 'LINES', {"pos": [left_top, left_bottom]}).draw(shader)
            center = Vector(((left_top[0]+right[0]+bottom[0]+left_bottom[0])*0.25,
                             (left_top[1]+right[1]+bottom[1]+left_bottom[1])*0.25))
            arrow_size = max(3.0, min(8.0, 5.0*zoom_scale))
            arrow_pts = [(center.x-arrow_size, center.y+arrow_size*0.35),
                         (center.x, center.y-arrow_size*0.55),
                         (center.x+arrow_size, center.y+arrow_size*0.35)]
            batch_for_shader(shader, 'LINE_STRIP', {"pos": arrow_pts}).draw(shader)

        if t_int == 1 and drawn_image and not node.text_content:
            draw_canvas_element_line(shader, node, shape_r, zoom_scale)
            return

        label_txt = canvas_preview_label(node)
        if t_int == 6:
            label_txt = ("☑ " if simulation_value(node,"checked",node.preview_checked) else "☐ ") + label_txt
        elif t_int == 8:
            options = [line.strip() for line in simulation_options(node) if line.strip()]
            if options:
                label_txt = options[min(simulation_value(node,"index",node.preview_combo_index), len(options) - 1)]
        if not label_txt:
            draw_canvas_element_line(shader, node, shape_r, zoom_scale)
            return

        elem_font_id = get_blf_font_id(node) if t_int in TEXT_CAPABLE_TYPES else 0
        base_font_sz = float(canvas_node_font_size(node, ntree)) if t_int in TEXT_CAPABLE_TYPES else 14.0
        font_px = max(1, round(base_font_sz * zoom_scale))
        blf.size(elem_font_id, font_px)
        _, txt_h = canvas_text_dimensions(elem_font_id, font_px, label_txt)

        elem_r_w = math.hypot(corners_r[1][0] - corners_r[0][0], corners_r[1][1] - corners_r[0][1])
        elem_r_h = math.hypot(corners_r[3][0] - corners_r[0][0], corners_r[3][1] - corners_r[0][1])

        pad_x = (canvas_text_padding(node) if t_int in TEXT_CAPABLE_TYPES else 8) * zoom_scale
        align_mode = int(node.alignment)
        local_ty = (elem_r_h - txt_h) * 0.5
        _, angle = canvas_element_pose(node, ntree)
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        if t_int == 0:
            label_txt = normalize_text_lines(label_txt)
            if not (node.multiline or node.wrap_text):
                label_txt = label_txt.replace("\n", " ")

        lines = normalize_text_lines(label_txt).split("\n") if t_int == 20 or (t_int == 0 and node.multiline) else [label_txt]
        if t_int == 0 and node.wrap_text:
            limit = min(elem_r_w, node.wrap_width * zoom_scale) if node.wrap_width else elem_r_w
            lines = canvas_wrapped_text(elem_font_id, font_px, label_txt, max(1, limit - 2 * pad_x))
        font_height = canvas_text_dimensions(elem_font_id, font_px, "Ag")[1]
        line_step = max(font_px, font_height)
        glyph_height = font_height if len(lines) > 1 else txt_h
        line_span = (len(lines) - 1) * line_step
        top, bottom = (value * zoom_scale for value in node.text_padding_vertical)
        vertical = node.text_vertical_align if t_int in TEXT_LAYOUT_TYPES else 'MIDDLE'
        top_y = elem_r_h - top - glyph_height
        free = max(0, top_y - bottom - line_span)
        first_y = top_y - free * {'TOP': 0, 'MIDDLE': .5, 'BOTTOM': 1}[vertical]

        blf.color(elem_font_id, *display_rgba(canvas_control_text_color(node, t_int)))
        clip_text = t_int in {0, 20} and abs(math.remainder(angle, 2.0 * math.pi)) < 1e-5
        if clip_text:
            blf.clipping(elem_font_id, min(p[0] for p in corners_r), min(p[1] for p in corners_r),
                         max(p[0] for p in corners_r), max(p[1] for p in corners_r))
            blf.enable(elem_font_id, blf.CLIPPING)
        try:
            for index, line_text in enumerate(lines):
                line_width, _ = canvas_text_dimensions(elem_font_id, font_px, line_text)
                if align_mode == 1:
                    line_x = (elem_r_w - line_width) * 0.5
                elif align_mode == 2:
                    line_x = elem_r_w - line_width - pad_x
                else:
                    line_x = pad_x
                line_y = first_y - index * line_step
                tx = round(corners_r[0][0] + line_x * cos_a - line_y * sin_a)
                ty = round(corners_r[0][1] + line_x * sin_a + line_y * cos_a)
                blf.position(elem_font_id, tx, ty, 0)
                blf.draw(elem_font_id, line_text)
        finally:
            if clip_text:
                blf.disable(elem_font_id, blf.CLIPPING)

        if t_int == 8 and simulation_value(node,"open",node.preview_combo_open):
            # Preview dropdown list (non-destructive: not serialized into Armory JSON).
            options = [v.strip() for v in simulation_options(node) if v.strip()]
            if options:
                top_left, top_right, bottom_right, bottom_left = corners_r
                step_x = bottom_left[0] - top_left[0]
                step_y = bottom_left[1] - top_left[1]
                for idx, option in enumerate(options[:24]):
                    shift = idx + 1
                    quad = [(top_left[0]+step_x*shift, top_left[1]+step_y*shift),
                            (top_right[0]+step_x*shift, top_right[1]+step_y*shift),
                            (bottom_right[0]+step_x*shift, bottom_right[1]+step_y*shift),
                            (bottom_left[0]+step_x*shift, bottom_left[1]+step_y*shift)]
                    shader.bind()
                    shader.uniform_float("color", display_rgba(node.color_hover) if idx == simulation_value(node,"index",node.preview_combo_index) and node.use_custom_button_states else ((0.36, 0.57, 0.76, 0.98) if idx == simulation_value(node,"index",node.preview_combo_index) else (0.22, 0.24, 0.27, 0.98)))
                    batch_for_shader(shader, 'TRI_FAN', {"pos": quad}).draw(shader)
                    blf.color(elem_font_id, 1.0, 1.0, 1.0, 1.0)
                    blf.position(elem_font_id, round(quad[3][0]+pad_x), round(quad[3][1]+local_ty), 0)
                    blf.draw(elem_font_id, option)
        draw_canvas_element_line(shader, node, shape_r, zoom_scale)


    for node in draw_order:
        if not node.is_effectively_visible() or not clip_node(node):
            continue
        if not canvas_has_offset(node):
            draw_element(node)
            continue
        _, center, _ = get_element_corners_view(node, ntree)
        x, y = to_reg(*center)
        dx, dy = node.offset_position
        zoom_x = (to_reg(center[0] + 100, center[1])[0] - x) / 100
        zoom_y = (to_reg(center[0], center[1] + 100)[1] - y) / 100
        with gpu.matrix.push_pop():
            gpu.matrix.multiply_matrix(
                Matrix.Translation((x + dx * zoom_x, y - dy * zoom_y, 0))
                @ Matrix.Rotation(-node.offset_rotation, 4, 'Z')
                @ Matrix.Diagonal((*node.offset_scale_xy, 1, 1))
                @ Matrix.Translation((-x, -y, 0)))
            draw_element(node)

    for node in draw_order:
        if not canvas_scroll_enabled(node) or not node.is_effectively_visible() or not clip_node(node):
            continue
        for _, track, thumb, _, _ in canvas_scroll_bars(ntree, node):
            shader.bind()
            for bounds, color in ((track, display_rgba(node.scroll_track_color)), (thumb, display_rgba(node.scroll_bar_color))):
                shader.uniform_float("color", color)
                points = [to_reg(*point) for point in canvas_box_points(bounds)]
                batch_for_shader(shader, 'TRI_FAN', {"pos": points}).draw(shader)
    apply_clip(None)

    if (ntree.show_element_names and isinstance(active_node, ArmoryCanvasNodeBase)
            and active_node.select and active_node.is_effectively_visible()
            and active_node.element_name and clip_node(active_node)):
        font = 0
        corners, _, _ = get_element_corners_view(active_node, ntree)
        points = [to_reg(*corner) for corner in corners]
        x = min(point[0] for point in points)
        y = max(point[1] for point in points) + 4
        blf.size(font, 12)
        name = active_node.element_name
        if (x <= region.width + 4 and -20 <= y <= region.height + 16
                and (x >= 0 or x + canvas_text_dimensions(font, 12, name)[0] >= -4)):
            blf.enable(font, blf.SHADOW)
            try:
                blf.shadow(font, 3, 0.0, 0.0, 0.0, 1.0)
                blf.shadow_offset(font, 1, -1)
                blf.color(font, 1.0, 1.0, 1.0, 1.0)
                blf.position(font, x, y, 0)
                blf.draw(font, name)
            finally:
                blf.disable(font, blf.SHADOW)
    apply_clip(None)

    gpu.state.blend_set('ALPHA')
    shader.bind()
    selected, scalable, scale_quad = canvas_scale_selection(ntree) if show_gizmos else ((), (), ())
    multi_scale = len(selected) > 1
    if (show_gizmos and not multi_scale and isinstance(active_node, ArmoryCanvasNodeBase)
            and active_node.select and not element_selection_locked(active_node)
            and active_node.is_effectively_visible() and clip_node(active_node)):
        corners_v, center_v, rot_v = get_element_corners_view(active_node, ntree)
        corners_r = [snap_px(to_reg(x, y)) for (x, y) in corners_v]
        center_r = snap_px(to_reg(*center_v))

        ax, ay = active_node.get_anchor_origin(ntree)
        anc_r = snap_px_half(to_reg(ax, ch - ay))
        tl_elem_r = snap_px_half(corners_r[3])
        batch_anc = batch_for_shader(shader, 'LINES', {"pos": [anc_r, tl_elem_r]})
        shader.uniform_float("color", (0.2, 0.7, 1.0, 0.85))
        gpu.state.line_width_set(1.0)
        batch_anc.draw(shader)
        draw_square_handle(shader, anc_r, 3.5, (0.2, 0.8, 1.0, 1.0), (0.05, 0.25, 0.4, 1.0))

        if not (active_node.group_locked and node_has_children(active_node)):
            rot_r = snap_px(to_reg(*rot_v))
            top_mid_r = snap_px_half((
                (corners_r[2][0] + corners_r[3][0]) * 0.5,
                (corners_r[2][1] + corners_r[3][1]) * 0.5
            ))
            batch_stem = batch_for_shader(shader, 'LINES', {"pos": [top_mid_r, snap_px_half(rot_r)]})
            shader.uniform_float("color", (0.1, 0.85, 0.3, 1.0))
            batch_stem.draw(shader)

            for cr in corners_r if any(canvas_resize_axes(active_node)) else ():
                draw_square_handle(shader, cr, 5.0, (1.0, 1.0, 1.0, 1.0), (1.0, 0.45, 0.0, 1.0))

            draw_square_handle(shader, rot_r, 5.0, (0.2, 0.95, 0.4, 1.0), (0.05, 0.35, 0.1, 1.0))
            draw_square_handle(shader, center_r, 4.0, (1.0, 0.75, 0.0, 0.95), (0.2, 0.2, 0.2, 1.0))

        x, y = snap_px_half(to_reg(*canvas_pivot_view(active_node, ntree)))
        cross = [(x - 7, y), (x + 7, y), (x, y - 7), (x, y + 7)]
        batch = batch_for_shader(shader, 'LINES', {"pos": cross})
        gpu.state.line_width_set(4.0)
        shader.uniform_float("color", (0.12, 0.02, 0.02, 1.0))
        batch.draw(shader)
        gpu.state.line_width_set(2.0)
        shader.uniform_float("color", (1.0, 0.12, 0.12, 1.0))
        batch.draw(shader)
        gpu.state.line_width_set(1.0)

    apply_clip(None)
    draw_canvas_selection_overlays(shader, region)

    if show_gizmos:
        if scale_quad:
            draw_canvas_scale_controls(shader, ntree, v2d, scale_quad, multi_scale, scalable)
        target = parent_gizmo_position(ntree, active_node, v2d)
        if target is not None:
            _, point = target
            draw_square_handle(shader, point, 6.0, (0.48, 0.25, 0.9, 1.0), (0.19, 0.07, 0.4, 1.0))
        for node, center in canvas_group_badges(ntree, v2d, draw_order):
            draw_group_lock_badge(shader, center, node.children_locked or node.group_locked, node.group_locked)
        center = snap_px(to_reg(cw, bottom))
        draw_square_handle(shader, center, 6.0, (0.2, 0.8, 1.0, 1.0), (0.05, 0.25, 0.4, 1.0))


CANVAS_SCALE_COLOR = (1.0, 0.55, 0.0, 1.0)


def canvas_resize_axes(node):
    parent = active_layout_parent(node)
    if parent is None or parent.layout_use_child_scale and parent.radio_layout_mode != 'GRID':
        return True, True
    if parent.radio_layout_mode == 'GRID':
        return False, False
    return tuple(not controlled for controlled in parent.layout_control_size)


def canvas_scale_selection(tree):
    selected = [node for node in canvas_selected_nodes(tree)
                if node.is_effectively_visible() and not node.group_locked]
    if not selected:
        return (), (), ()
    if len(selected) > 1:
        quad = canvas_box_points(canvas_union_bounds([
            canvas_bounds(get_element_corners_view(node, tree)[0]) for node in selected]))
    else:
        quad = get_element_corners_view(selected[0], tree)[0]
    movable = [node for node in selected if any(canvas_resize_axes(node))
               and not any(parent != node and is_ancestor_of(parent, node) for parent in selected)]
    return selected, movable, tuple(map(Vector, quad))


def canvas_scale_controls(quad, view):
    corners = [Vector(view.view_to_region(*point, clip=False)) for point in quad]
    edges = [(corners[i] + corners[(i + 1) % 4]) * .5 for i in range(4)]
    top = edges[2]
    circles = [top + Vector((offset, 18)) for offset in (-10, 10)]
    return corners, edges, circles


def draw_canvas_scale_controls(shader, tree, view, quad, multi, scalable):
    corners, edges, circles = canvas_scale_controls(quad, view)
    shader.uniform_float("color", CANVAS_SCALE_COLOR)
    batch_for_shader(shader, 'LINE_STRIP', {"pos": corners + corners[:1]}).draw(shader)
    if multi and scalable:
        for corner in corners:
            draw_square_handle(shader, corner, 4, (1, 1, 1, 1), CANVAS_SCALE_COLOR)
    lines = ([], [])
    editable = [any(canvas_resize_axes(node)[axis] for node in scalable) for axis in range(2)]
    for i, center in enumerate(edges):
        direction = corners[(i + 1) % 4] - corners[i]
        if direction.length > 0:
            half = direction.normalized() * min(7, direction.length * .25)
            lines[int(editable[1 - i % 2])].extend((center - half, center + half))
    for points, color in zip(lines, ((.65, .65, .65, .8), CANVAS_SCALE_COLOR)):
        if points:
            shader.uniform_float("color", color)
            gpu.state.line_width_set(3)
            batch_for_shader(shader, 'LINES', {"pos": points}).draw(shader)
            gpu.state.line_width_set(1)
    if multi:
        for mode, center, color in zip((True, False), circles,
                                       ((.68, .38, 1, 1), (.27, .9, .44, 1))):
            active = tree.proportional_gizmo_scale == mode
            ring = [center + Vector((math.cos(i * math.tau / 16), math.sin(i * math.tau / 16)))
                    * (5 if active else 4) for i in range(16)]
            shader.uniform_float("color", color)
            batch_for_shader(shader, 'TRI_FAN', {"pos": ring}).draw(shader)
            if active:
                shader.uniform_float("color", (1, 1, 1, 1))
                batch_for_shader(shader, 'LINE_STRIP', {"pos": ring + ring[:1]}).draw(shader)


def canvas_reposition_visual_center(tree, node, target):
    center, angle = canvas_element_pose(node, tree)
    delta = Vector(target) - center
    parent_angle = angle + node.rotation
    local = transform_point(delta.x, delta.y, Vector((0, 0)),
                            math.cos(parent_angle), -math.sin(parent_angle))
    set_canvas_transform(node, pos_x=node.pos_x + round(local[0]), pos_y=node.pos_y - round(local[1]))


def canvas_selected_nodes(tree):
    return [node for node in get_ordered_canvas_nodes(tree)
            if node.select and not element_selection_locked(node)]


def activate_canvas_node(tree, node, *, exclusive=True):
    if exclusive:
        for other in tree.nodes:
            other.select = False
    tree.nodes.active = node
    if node is not None:
        node.select = True
        sync_active_element_index(tree, node)


def parent_gizmo_position(tree, node, v2d):
    if not isinstance(node, ArmoryCanvasNodeBase) or not node.select or not node.is_effectively_visible():
        return None
    parent = get_parent_node(node)
    if parent is None or element_selection_locked(parent):
        return None
    corners, _, _ = get_element_corners_view(node, tree)
    x, y = v2d.view_to_region(*corners[3], clip=False)
    return parent, (x - 16, y + 12)


# -------------------------------------------------------------------------
# 5. Interactive Modal Gizmo Operator
# -------------------------------------------------------------------------
_canvas_guides = {}
_canvas_marquees = {}


def canvas_bounds(points):
    return tuple((min(point[axis] for point in points), max(point[axis] for point in points))
                 for axis in range(2))


def canvas_union_bounds(boxes):
    return tuple((min(box[axis][0] for box in boxes), max(box[axis][1] for box in boxes))
                 for axis in range(2))


def canvas_box_points(bounds):
    (x0, x1), (y0, y1) = bounds
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def canvas_intersect_bounds(a, b):
    return tuple((max(a[axis][0], b[axis][0]), min(a[axis][1], b[axis][1])) for axis in range(2))


def canvas_quad_intersects_box(quad, bounds):
    box = canvas_box_points(bounds)
    for i in (0, 1):
        x0, y0 = quad[i]
        x1, y1 = quad[i + 1]
        normal = (y1 - y0, x0 - x1)
        first = [x * normal[0] + y * normal[1] for x, y in quad]
        second = [x * normal[0] + y * normal[1] for x, y in box]
        if max(first) < min(second) or max(second) < min(first):
            return False
    return True


def canvas_art_bounds(tree):
    width, height = canvas_preview_size(tree)
    top = canvas_view_top(tree)
    return ((0, width), (top - height, top))


def canvas_movable_selection(tree):
    selected = [node for node in canvas_selected_nodes(tree) if node.is_effectively_visible()
                and not node.group_locked and active_layout_parent(node) is None]
    return [node for node in selected if not any(
        parent != node and is_ancestor_of(parent, node) for parent in selected)]


def canvas_snap_targets(tree, moving):
    moved = set(moving)
    parents = {get_parent_node(node) for node in moved}
    sources = [canvas_art_bounds(tree)]
    for node in get_ordered_canvas_nodes(tree):
        if (node in moved or not node.is_effectively_visible()
                or any(is_ancestor_of(ancestor, node) for ancestor in moved)):
            continue
        if node in parents or get_parent_node(node) in parents:
            sources.append(canvas_bounds(get_element_corners_view(node, tree)[0]))
    return sources


def set_canvas_transform(node, **values):
    changed = False
    for name, value in values.items():
        if name in ('width', 'height'):
            value = max(1, int(value))
            bit = 1 if name == 'width' else 2
            if node.get(name) == value:
                if not node.get('_stretch_mode', 0) & bit or active_layout_parent(node) is not None:
                    continue
                offset = value - canvas_container_size(node, name, {node})
                if node.get('_stretch_' + name, 0) == offset:
                    continue
        else:
            old = getattr(node, name)
            if old == value or (name == 'rotation' and math.isclose(old, value, rel_tol=0.0, abs_tol=1e-6)):
                continue
        setattr(node, name, value)
        changed = True
    return changed


def canvas_alignment_snap(tree, region, bounds, delta, targets, axis=None, bypass=False):
    key = region.as_pointer()
    _canvas_guides.pop(key, None)
    if not tree.smart_snap or bypass:
        return delta, (False, False)
    view = region.view2d
    start, end = view.region_to_view(0, 0), view.region_to_view(1, 1)
    moving = tuple((low + delta[i], high + delta[i]) for i, (low, high) in enumerate(bounds))
    corrections = [0.0, 0.0]
    guides = []
    snapped = [False, False]
    for dimension, label in enumerate(('X', 'Y')):
        if axis is not None and axis != label:
            continue
        threshold = abs(end[dimension] - start[dimension]) * tree.smart_snap_distance
        low, high = moving[dimension]
        anchors = (low, (low + high) * 0.5, high)
        best = None
        for target in targets:
            low, high = target[dimension]
            for match in (low, (low + high) * 0.5, high):
                for value in anchors:
                    difference = match - value
                    if abs(difference) <= threshold and (best is None or abs(difference) < abs(best[0])):
                        best = difference, match, target
        if best is not None:
            corrections[dimension], match, target = best
            snapped[dimension] = True
            other = 1 - dimension
            low = min(moving[other][0], target[other][0])
            high = max(moving[other][1], target[other][1])
            guides.append((dimension, match, low, high))
    _canvas_guides[key] = guides
    return delta + Vector(corrections), snapped


def canvas_snap_scale_factor(tree, region, bounds, pivot, factor, targets, axis=None, bypass=False):
    """Choose the closest reachable edge or center while preserving the scale pivot."""
    key = region.as_pointer()
    _canvas_guides.pop(key, None)
    if not tree.smart_snap or bypass:
        return factor, False
    view = region.view2d
    origin, pixel = view.region_to_view(0, 0), view.region_to_view(1, 1)
    best = None
    for i, label in enumerate(('X','Y')):
        if axis and axis != label:
            continue
        size = max(1e-9, abs(pixel[i] - origin[i]))
        lo, hi = bounds[i]
        for anchor in (lo, (lo+hi)/2, hi):
            span = anchor - pivot[i]
            if abs(span) < .001:
                continue
            for target in targets:
                low, high = target[i]
                for guide in (low, (low+high)/2, high):
                    candidate = (guide - pivot[i]) / span
                    distance = abs((candidate-factor)*span) / size
                    if candidate >= .01 and distance <= tree.smart_snap_distance and (best is None or distance < best[0]):
                        best = distance, candidate, i, guide, target
    if best is None:
        return factor, False
    _, factor, i, guide, target = best
    j = 1-i
    extents = [pivot[j]+(v-pivot[j])*(1 if axis else factor) for v in bounds[j]]
    _canvas_guides[key] = [(i, guide, min(extents[0], target[j][0]), max(extents[1], target[j][1]))]
    return factor, True


def draw_canvas_selection_overlays(shader, region):
    view = region.view2d
    key = region.as_pointer()
    guide_points = []
    for axis, position, low, high in _canvas_guides.get(key, ()):
        points = ((position, low), (position, high)) if axis == 0 else ((low, position), (high, position))
        guide_points.extend(snap_px_half(view.view_to_region(*point, clip=False)) for point in points)
    if guide_points:
        shader.bind()
        shader.uniform_float("color", (0.95, 0.18, 0.65, 1.0))
        batch_for_shader(shader, 'LINES', {"pos": guide_points}).draw(shader)
    if key in _canvas_marquees:
        points = canvas_box_points(canvas_bounds(_canvas_marquees[key]))
        shader.bind()
        gpu.state.line_width_set(1.0)
        shader.uniform_float("color", (0.41, 0.76, 1.0, 0.95))
        batch_for_shader(shader, 'LINE_STRIP', {"pos": points + points[:1]}).draw(shader)


class NODE_OT_armory_canvas_marquee(Operator):
    bl_idname = "node.armory_canvas_marquee"
    bl_label = "Box Select Canvas Elements"
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        tree = canvas_shortcut_tree(context)
        return tree is not None and tree.show_canvas_overlay and context.region.type == 'WINDOW'

    def invoke(self, context, event):
        self.key = context.region.as_pointer()
        self.start = (event.mouse_region_x, event.mouse_region_y)
        self.drag = event.type == 'LEFTMOUSE'
        self.extend = event.shift
        self.subtract = event.ctrl
        self.started = self.drag
        if self.started:
            _canvas_marquees[self.key] = (self.start, self.start)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        point = (event.mouse_region_x, event.mouse_region_y)
        if event.type in {'ESC', 'RIGHTMOUSE'}:
            self.cancel(context)
            return {'CANCELLED'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and not self.started:
            self.started = True
            self.start = point
            self.extend |= event.shift
            self.subtract |= event.ctrl
        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.started:
            self.select(context, point)
            self.cancel(context)
            return {'FINISHED'}
        if self.started:
            _canvas_marquees[self.key] = self.start, point
            context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def select(self, context, point):
        tree = canvas_shortcut_tree(context)
        view = context.region.view2d
        selection = canvas_bounds([view.region_to_view(*p) for p in (self.start, point)])
        if not (self.extend or self.subtract):
            activate_canvas_node(tree, None)
        if max(abs(a - b) for a, b in zip(self.start, point)) < 3:
            return
        hits = []
        for node in canvas_paint_order(tree):
            if not node.is_effectively_visible() or element_selection_locked(node):
                continue
            bounds = canvas_visible_bounds(tree, node)
            if bounds is None:
                continue
            overlap = canvas_intersect_bounds(selection, bounds)
            if all(low <= high for low, high in overlap):
                quad, _, _ = get_element_corners_view(node, tree)
                if canvas_quad_intersects_box(quad, overlap):
                    node.select = not self.subtract
                    hits.append(node)
        active = tree.nodes.active
        if hits and not self.subtract:
            activate_canvas_node(tree, hits[-1], exclusive=False)
        elif active is not None and not active.select:
            selected = canvas_selected_nodes(tree)
            activate_canvas_node(tree, selected[-1] if selected else None, exclusive=False)

    def cancel(self, context):
        _canvas_marquees.pop(self.key, None)
        context.area.tag_redraw()


class NODE_OT_armory_canvas_frame(Operator):
    bl_idname = "node.armory_canvas_frame"
    bl_label = "Frame Canvas"

    selected: BoolProperty(default=False)

    @classmethod
    def poll(cls, context):
        tree = canvas_shortcut_tree(context)
        return tree is not None and tree.show_canvas_overlay and context.region.type == 'WINDOW'

    def execute(self, context):
        tree = canvas_shortcut_tree(context)
        bounds = canvas_art_bounds(tree)
        if self.selected:
            visible = [canvas_visible_bounds(tree, node) for node in canvas_selected_nodes(tree)
                       if node.is_effectively_visible()]
            visible = [box for box in visible if box is not None]
            if not visible:
                return {'PASS_THROUGH'}
            bounds = canvas_union_bounds(visible)
        (xmin, xmax), (ymin, ymax) = bounds
        margin = max(24, (xmax - xmin) * .08, (ymax - ymin) * .08)
        side_width = next((region.width for region in context.area.regions
                           if region.type == 'UI' and region.width > 1), 0)
        width = max(1, context.region.width - side_width)
        ratio = width / max(1, context.region.height)
        half_x = max((xmax - xmin) / 2 + margin, ((ymax - ymin) / 2 + margin) * ratio)
        half_y = half_x / ratio
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        intent_x = cx + half_x * side_width / width
        extent_x = half_x * context.region.width / width
        view = context.region.view2d
        minimum = view.view_to_region(intent_x - extent_x, cy - half_y, clip=False)
        maximum = view.view_to_region(intent_x + extent_x, cy + half_y, clip=False)
        return bpy.ops.view2d.zoom_border(xmin=minimum[0], xmax=maximum[0],
                                          ymin=minimum[1], ymax=maximum[1], zoom_out=False)


CANVAS_ALIGNMENT_ACTIONS = (
    ('LEFT', "Left", 'ALIGN_LEFT'), ('CENTER', "Center", 'ALIGN_CENTER'),
    ('RIGHT', "Right", 'ALIGN_RIGHT'), ('TOP', "Top", 'TRIA_UP'),
    ('MIDDLE', "Middle", 'REMOVE'), ('BOTTOM', "Bottom", 'TRIA_DOWN'),
    ('HORIZONTAL', "Horizontal Distribution", 'ARROW_LEFTRIGHT'),
    ('VERTICAL', "Vertical Distribution", 'SORT_DESC'),
)


class NODE_OT_armory_canvas_align_selection(Operator):
    bl_idname = "node.armory_canvas_align_selection"
    bl_label = "Align or Distribute Selection"
    bl_options = {'REGISTER', 'UNDO'}
    action: EnumProperty(items=[(key, label, label) for key, label, _ in CANVAS_ALIGNMENT_ACTIONS])

    def execute(self, context):
        tree = canvas_shortcut_tree(context)
        if tree is None:
            return {'CANCELLED'}
        nodes = canvas_movable_selection(tree)
        boxes = {node: canvas_bounds(get_element_corners_view(node, tree)[0]) for node in nodes}
        distribute = self.action in {'HORIZONTAL', 'VERTICAL'}
        if len(nodes) < (3 if distribute else 2):
            self.report({'WARNING'}, "Select movable elements outside automatic layout")
            return {'CANCELLED'}
        axis = 0 if self.action in {'LEFT', 'CENTER', 'RIGHT', 'HORIZONTAL'} else 1
        bounds = canvas_union_bounds(boxes.values())
        low, high = bounds[axis]
        shifts = {}
        if distribute:
            nodes.sort(key=lambda node: sum(boxes[node][axis]))
            low, high = boxes[nodes[0]][axis][0], boxes[nodes[-1]][axis][1]
            gap = (high - low - sum(boxes[node][axis][1] - boxes[node][axis][0] for node in nodes)) / (len(nodes) - 1)
            cursor = low
            for previous, node in zip(nodes, nodes[1:-1]):
                a, b = boxes[previous][axis]
                cursor += b - a + gap
                shifts[node] = cursor - boxes[node][axis][0]
        else:
            factor = {'LEFT': 0, 'BOTTOM': 0, 'CENTER': .5, 'MIDDLE': .5, 'RIGHT': 1, 'TOP': 1}[self.action]
            target = low + (high - low) * factor
            for node in nodes:
                a, b = boxes[node][axis]
                shifts[node] = target - (a + (b - a) * factor)
        for node, shift in shifts.items():
            if axis == 0:
                node.pos_x += round(shift)
            else:
                node.pos_y -= round(shift)
        context.area.tag_redraw()
        return {'FINISHED'}


_PREFAB_RAW_KEYS = ('width', 'height', '_stretch_mode', '_stretch_width', '_stretch_height',
                    '_armory_canvas_original_element', '_armory_imported_asset_path')
_PREFAB_EXCLUDED = {'width', 'height', 'z_order', 'stretch_mode', 'asset_image', 'button_image', 'multiline_text',
                    'preview_scroll', 'offset_transform_expanded'}


def canvas_prefab_properties():
    return [key for key in ArmoryCanvasNodeBase.__annotations__ if key not in _PREFAB_EXCLUDED]


def canvas_prefab_document(tree):
    selected = set(canvas_selected_nodes(tree))
    nodes = [node for node in get_ordered_canvas_nodes(tree)
             if node in selected or any(is_ancestor_of(root, node) for root in selected)]
    indices = {node: index for index, node in enumerate(nodes)}
    entries = []
    if not nodes:
        return {'version': 1, 'elements': entries}
    roots = [node for node in nodes if get_parent_node(node) not in indices]
    locations = {node: node.get_world_pos(tree) for node in roots}
    origin_x = min(location[0] for location in locations.values())
    origin_y = min(location[1] for location in locations.values())
    for node in nodes:
        props = {}
        for key in canvas_prefab_properties():
            value = getattr(node, key)
            props[key] = list(value) if getattr(node.bl_rna.properties[key], 'is_array', False) else value
        if props['asset_path'] and props['asset_type'] != 'NONE':
            props['asset_path'] = os.path.abspath(bpy.path.abspath(props['asset_path']))
        if props['button_image_path']:
            props['button_image_path'] = os.path.abspath(bpy.path.abspath(props['button_image_path']))
        parent = indices.get(get_parent_node(node))
        if parent is None:
            x, y = locations[node]
            props.update(anchor='0', pos_x=x - origin_x, pos_y=y - origin_y)
        raw = {key: node[key] for key in _PREFAB_RAW_KEYS if key in node}
        entries.append({'node': node.bl_idname, 'parent': parent, 'properties': props, 'raw': raw})
    return {'version': 1, 'elements': entries}


def place_canvas_prefab(tree, document):
    if not isinstance(document, dict) or document.get('version') != 1 or not isinstance(document.get('elements'), list):
        raise ValueError("Unsupported Canvas prefab")
    entries = document['elements']
    node_types = set(TYPE_TO_NODE_BL_IDNAME.values())
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get('node') not in node_types or not all(
                isinstance(entry.get(key), dict) for key in ('properties', 'raw')):
            raise ValueError("Invalid prefab element")
        seen = {index}
        parent = entry.get('parent')
        while parent is not None:
            if type(parent) is not int or not 0 <= parent < len(entries) or parent in seen:
                raise ValueError("Invalid prefab hierarchy")
            seen.add(parent)
            ancestor = entries[parent]
            if not isinstance(ancestor, dict):
                raise ValueError("Invalid prefab hierarchy")
            parent = ancestor.get('parent')
    if not entries:
        raise ValueError("The prefab has no elements")
    prior_dirty = tree.is_dirty
    prior_active = tree.nodes.active
    prior_selection = [node for node in tree.nodes if node.select]
    nodes = []
    next_order = max((node.z_order for node in get_ordered_canvas_nodes(tree)), default=-1) + 1
    fields = set(canvas_prefab_properties())
    try:
        for index, entry in enumerate(entries):
            node = tree.nodes.new(entry['node'])
            nodes.append(node)
            node.z_order = next_order + index
            node.location = (canvas_preview_size(tree)[0] + 120 + (index % 3) * 200,
                             canvas_view_top(tree) - (index // 3) * 90)
            for key, value in entry['properties'].items():
                if key in fields and key != 'asset_name':
                    setattr(node, key, value)
            if 'asset_name' in entry['properties']:
                node.asset_name = entry['properties']['asset_name']
        for index, entry in enumerate(entries):
            node = nodes[index]
            parent = entry.get('parent')
            if parent is not None:
                if not set_node_parent(tree, node, nodes[parent]):
                    raise ValueError("Unable to restore prefab hierarchy")
            else:
                node.pos_x += 24
                node.pos_y += 24
            for key, value in entry['raw'].items():
                if key in _PREFAB_RAW_KEYS:
                    node[key] = value
    except (RuntimeError, TypeError, ValueError, OverflowError):
        for node in reversed(nodes):
            tree.nodes.remove(node)
        for node in tree.nodes:
            node.select = node in prior_selection
        tree.nodes.active = prior_active
        tree.is_dirty = prior_dirty
        raise
    roots = [node for node, entry in zip(nodes, entries) if entry.get('parent') is None]
    activate_canvas_node(tree, roots[0])
    for node in roots:
        node.select = True
    tree.is_dirty = True
    return len(nodes)


_canvas_simulations = {}


def simulation_value(node, key, default):
    session = _canvas_simulations.get(node.id_data.as_pointer())
    if session is None:
        return default
    return session['values'].get((node.name, key), default)


def simulation_hits(tree, state, point):
    global _canvas_draw_cache
    previous = _canvas_draw_cache
    _canvas_draw_cache = canvas_geometry_cache(tree)
    try:
        if 'hit_index' not in state:
            cells, large = {}, []
            for rank, node in enumerate(canvas_paint_order(tree)):
                if not (canvas_scroll_enabled(node) or int(node.armory_type) in {1,2,6,7,8,9,10,11,20}):
                    continue
                if not node.is_effectively_visible():
                    continue
                bounds = canvas_visible_bounds(tree, node, interaction=True)
                if bounds is None:
                    continue
                (x0,x1),(y0,y1) = [(math.floor(lo/128),math.floor(hi/128)) for lo,hi in bounds]
                entry = (rank, node)
                if (x1-x0+1)*(y1-y0+1) > 64:
                    large.append(entry)
                else:
                    for x in range(x0,x1+1):
                        for y in range(y0,y1+1):
                            cells.setdefault((x,y), []).append(entry)
            state['hit_index'] = cells, large
        cells, large = state['hit_index']
        local = cells.get(tuple(math.floor(v/128) for v in point), ())
        candidates = reversed(sorted((*local, *large))) if large else reversed(local)
        return [node for _,node in candidates if canvas_point_in_visible_node(tree, node, point, interaction=True)]
    finally:
        _canvas_draw_cache = previous


def simulation_uv(tree, node, point):
    corners = canvas_interaction_corners(tree, node)
    offset = Vector(point) - Vector(corners[0])
    axes = [Vector(corners[i]) - Vector(corners[0]) for i in (1, 3)]
    return tuple(offset.dot(axis) / max(.000001, axis.length_squared) for axis in axes)


def simulation_options(node):
    if node.id_data.as_pointer() in _canvas_simulations:
        return node.text_content.split(';') if node.text_content else node.preview_combo_items.splitlines() or [node.element_name]
    return node.preview_combo_items.splitlines()


class NODE_OT_armory_canvas_simulate(Operator):
    bl_idname = "node.armory_canvas_simulate"
    bl_label = "Preview"
    bl_description = "Interact with this canvas in Blender. Click again or press Escape to stop"

    def invoke(self, context, event):
        tree = canvas_shortcut_tree(context)
        if tree is None:
            return {'CANCELLED'}
        self.key = tree.as_pointer()
        previous = _canvas_simulations.pop(self.key, None)
        invalidate_canvas_geometry(tree)
        if previous is not None:
            context.area.tag_redraw()
            return {'FINISHED'}
        self.tree = tree
        self.area = context.area
        self.region = next(r for r in self.area.regions if r.type == 'WINDOW')
        self.state = {'values': {}, 'hover': '', 'pressed': '', 'event': ''}
        _canvas_simulations[self.key] = self.state
        self.focus = self.combo = self.slider = self.bar = None
        tree.show_canvas_overlay = True
        self.area.tag_redraw()
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def stop(self):
        if _canvas_simulations.get(self.key) is self.state:
            _canvas_simulations.pop(self.key, None)
            invalidate_canvas_geometry(self.tree)
        self.area.tag_redraw()
        return {'FINISHED'}

    def modal(self, context, event):
        if _canvas_simulations.get(self.key) is not self.state:
            return {'FINISHED'}
        if self.area.type != 'NODE_EDITOR' or self.area.spaces.active.edit_tree != self.tree:
            return self.stop()
        if event.type == 'WINDOW_DEACTIVATE' or (event.type == 'ESC' and event.value == 'PRESS'):
            return self.stop()
        region = self.region
        x, y = event.mouse_x - region.x, event.mouse_y - region.y
        if not (0 <= x < region.width and 0 <= y < region.height) and not (self.bar or self.slider):
            if self.state['hover']:
                self.state['hover'] = ''
                self.state['hover_end'] = 0
                region.tag_redraw()
            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                self.state['pressed'] = ''
            return {'PASS_THROUGH'}
        point = region.view2d.region_to_view(x, y)
        values = self.state['values']
        nodes = self.tree.nodes
        previous_event = self.state['event']
        focus = nodes.get(self.focus or '')
        if focus and event.value == 'PRESS' and event.type not in {'LEFTMOUSE', 'RIGHTMOUSE', 'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            key = (focus.name, 'input')
            value = values.get(key, focus.preview_input_text or focus.text_content)
            if focus.armory_type == '11':
                values[key] = event.type
                self.focus = None
            elif event.type == 'BACK_SPACE':
                values[key] = value[:-1]
            elif event.type in {'RET', 'NUMPAD_ENTER'}:
                if focus.armory_type == '20':
                    values[key] = value + '\n'
                else:
                    self.focus = None
            elif event.unicode and event.unicode.isprintable():
                values[key] = value + event.unicode
            self.state['event'] = focus.event_name
            if previous_event != focus.event_name:
                self.area.tag_redraw()
            else:
                region.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type not in {'LEFTMOUSE', 'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'RUNNING_MODAL'}
        previous_hover = self.state['hover'], self.state.get('hover_end', 0)
        hits = simulation_hits(self.tree, self.state, point)
        hit = next((node for node in hits if int(node.armory_type) in {1,2,6,7,8,9,10,11,20}), None)
        scrollers = [node for node in hits if canvas_scroll_enabled(node)]
        if hit and hit.disabled:
            hit = None
        self.state['hover'] = hit.name if hit else ''
        side = 0
        if hit and hit.control_variant == 'STEPPER':
            u, _ = simulation_uv(self.tree, hit, point)
            edge = min(hit.height / max(1, hit.width), 1/3)
            side = -1 if u <= edge else 1 if u >= 1-edge else 0
        self.state['hover_end'] = side
        if (event.type in {'MOUSEMOVE','INBETWEEN_MOUSEMOVE'} and not (self.bar or self.slider)
                and previous_hover == (self.state['hover'], side)):
            return {'RUNNING_MODAL'}
        if event.type in {'WHEELUPMOUSE','WHEELDOWNMOUSE'}:
            for node in scrollers:
                limits = canvas_scroll_range(node)
                axis = 0 if event.shift or not limits[1] else 1
                old = list(canvas_scroll_offset(node))
                value = min(limits[axis], max(0, old[axis] + (40 if event.type == 'WHEELDOWNMOUSE' else -40)))
                if value != old[axis]:
                    old[axis] = value
                    values[node.name, 'scroll'] = old
                    invalidate_canvas_geometry(self.tree)
                    break
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self.focus = None
            for node in scrollers:
                for axis, track, thumb, travel, limit in canvas_scroll_bars(self.tree, node):
                    if travel and all(lo <= point[i] <= hi for i,(lo,hi) in enumerate(track)):
                        self.bar = (node.name, axis, sum(thumb[axis])/2 if not thumb[axis][0] <= point[axis] <= thumb[axis][1] else point[axis],
                                    canvas_scroll_offset(node)[axis], limit/travel*(1 if axis==0 else -1), limit)
                        break
                if self.bar:
                    break
            if self.combo:
                combo = nodes.get(self.combo)
                self.combo = None
                if combo:
                    u,v = simulation_uv(self.tree, combo, point)
                    index = math.floor(-v)
                    clip = canvas_scroll_clip(self.tree, combo)
                    if (0 <= u <= 1 and v < 0 and 0 <= index < min(24,len(simulation_options(combo)))
                            and (clip is None or all(lo <= point[i] <= hi for i,(lo,hi) in enumerate(clip)))):
                        values[combo.name,'index'] = index
                        self.state['event'] = combo.event_name
                        hit = None
                    values[combo.name,'open'] = False
            self.state['pressed_end'] = side
            self.state['pressed'] = hit.name if hit and not self.bar else ''
            if hit and hit.armory_type == '9' and not self.bar:
                self.slider = hit.name
        if self.bar:
            name,axis,start,offset,factor,limit = self.bar
            node = nodes.get(name)
            if node:
                offsets = list(canvas_scroll_offset(node))
                offsets[axis] = max(0,min(limit,offset+(point[axis]-start)*factor))
                if offsets != list(canvas_scroll_offset(node)):
                    values[name,'scroll'] = offsets
                    invalidate_canvas_geometry(self.tree)
        if self.slider and nodes.get(self.slider):
            node = nodes[self.slider]
            values[node.name,'slider'] = min(1,max(0,simulation_uv(self.tree,node,point)[0]))
            self.state['event'] = node.event_name
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if hit and self.state['pressed'] == hit.name:
                kind = int(hit.armory_type)
                self.state['event'] = hit.event_name
                if kind == 6:
                    switch = hit.control_variant == 'SWITCH'
                    key = 'switch' if switch else 'checked'
                    values[hit.name,key] = not simulation_value(hit,key,hit.switch_initial if switch else hit.preview_checked)
                elif kind == 7:
                    for n in nodes:
                        if isinstance(n,ArmoryCanvasNodeBase) and n.armory_type == '7' and get_parent_node(n)==get_parent_node(hit):
                            values[n.name,'radio'] = n==hit
                elif kind == 8:
                    self.combo = hit.name
                    values[hit.name,'open'] = True
                elif kind in {10,11,20}:
                    self.focus = hit.name
                elif kind == 2 and hit.control_variant == 'STEPPER':
                    direction = side if side == self.state.get('pressed_end') else 0
                    values[hit.name,'step'] = max(hit.step_min,min(max(hit.step_min,hit.step_max),canvas_step_value(hit)+direction*hit.step_amount))
            self.slider = self.bar = None
            self.state['pressed'] = ''
        if previous_event != self.state['event']:
            self.area.tag_redraw()
        else:
            region.tag_redraw()
        return {'RUNNING_MODAL'}


def draw_canvas_alignment(layout, tree):
    count = len(canvas_movable_selection(tree))
    for first, last in ((0, 3), (3, 6)):
        row = layout.row(align=True)
        row.enabled = count >= 2
        for action, label, icon in CANVAS_ALIGNMENT_ACTIONS[first:last]:
            row.operator("node.armory_canvas_align_selection", text=label, icon=icon).action = action
    distribution = layout.column(align=True)
    distribution.enabled = count >= 3
    for action, label, icon in CANVAS_ALIGNMENT_ACTIONS[6:]:
        distribution.operator("node.armory_canvas_align_selection", text=label,
                              icon=icon).action = action


def draw_canvas_snapping(layout, tree):
    row = layout.row(align=True)
    row.prop(tree, "snap_to_grid", text="Grid")
    row.prop(tree, "grid_size", text="Size (px)")
    row = layout.row(align=True)
    row.prop(tree, "snap_rotation", text="Rotation")
    row.prop(tree, "rotation_snap_degrees", text="Degrees")
    layout.prop(tree, "snap_scale", text="Scale")
    row = layout.row(align=True)
    row.prop(tree, "smart_snap", text="Smart Alignment")
    row.prop(tree, "smart_snap_distance", text="Distance")


class CanvasHeaderPopover:
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'HEADER'

    @classmethod
    def poll(cls, context):
        return canvas_shortcut_tree(context) is not None


class NODE_PT_armory_canvas_header_snapping(CanvasHeaderPopover, Panel):
    bl_label = "Snapping"

    def draw(self, context):
        draw_canvas_snapping(self.layout, canvas_shortcut_tree(context))


class NODE_PT_armory_canvas_header_align(CanvasHeaderPopover, Panel):
    bl_label = "Align"

    def draw(self, context):
        draw_canvas_alignment(self.layout, canvas_shortcut_tree(context))


def draw_canvas_view_settings(layout, ntree):
    row = layout.row(align=True)
    row.prop(ntree, "show_canvas_overlay", text="Show Visual Canvas", toggle=True, icon='RESTRICT_VIEW_OFF')
    row.prop(ntree, "show_gizmos", text="Gizmos", toggle=True, icon='ORIENTATION_GLOBAL')
    layout.prop(ntree, "show_element_names", toggle=True)
    layout.prop(ntree, "show_viewport_fps", toggle=True)
    layout.prop(ntree, "color_code_nodes", toggle=True)
    if ntree.color_code_nodes:
        colors = (
            ("Basic", "node_color_basic"),
            ("Buttons", "node_color_buttons"),
            ("Inputs", "node_color_inputs"),
            ("Shapes", "node_color_shapes"),
            ("Progress Bars", "node_color_progress"),
        )
        col = layout.column(align=True)
        for label, prop in colors:
            row = col.split(factor=0.45, align=True)
            row.label(text=f"{label}:")
            row.prop(ntree, prop, text="")


class NODE_PT_armory_canvas_header_view(CanvasHeaderPopover, Panel):
    bl_label = "Show Visual Canvas"

    def draw(self, context):
        tree = canvas_shortcut_tree(context)
        draw_canvas_view_settings(self.layout, tree)
        self.layout.separator()
        draw_canvas_grid_settings(self.layout, tree)


def draw_canvas_simulation_header(self, context):
    tree = canvas_shortcut_tree(context)
    if tree is not None:
        row = self.layout.row(align=True)
        row.operator_context = 'INVOKE_DEFAULT'
        row.popover(panel="NODE_PT_armory_canvas_header_view", text="", icon='OVERLAY')
        row.popover(panel="NODE_PT_armory_canvas_header_snapping", text="Snapping", icon='SNAP_ON')
        row.popover(panel="NODE_PT_armory_canvas_header_align", text="Align", icon='ALIGN_CENTER')
        session = _canvas_simulations.get(tree.as_pointer())
        row.operator("node.armory_canvas_simulate", text="Preview",
                     icon='PAUSE' if session is not None else 'PLAY', depress=session is not None)
        row.operator("node.export_armory_canvas", text="", icon='EXPORT')
        row.operator("node.import_armory_canvas", text="", icon='IMPORT')
        if session and session['event']:
            row.label(text=session['event'])


class NODE_OT_armory_canvas_save_prefab(Operator, ExportHelper):
    bl_idname = "node.armory_canvas_save_prefab"
    bl_label = "Save Prefab"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    target_tree: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        tree = canvas_shortcut_tree(context)
        if tree is None:
            return {'CANCELLED'}
        self.target_tree = tree.name
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        tree = bpy.data.node_groups.get(self.target_tree) if self.target_tree else canvas_shortcut_tree(context)
        document = canvas_prefab_document(tree) if tree else {'elements': []}
        if not document['elements']:
            self.report({'WARNING'}, "Select a group or elements to save")
            return {'CANCELLED'}
        try:
            _canvas_atomic_write(bpy.path.abspath(self.filepath), json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        except (OSError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Saved {len(document['elements'])} prefab elements")
        return {'FINISHED'}


class NODE_OT_armory_canvas_place_prefab(Operator, ImportHelper):
    bl_idname = "node.armory_canvas_place_prefab"
    bl_label = "Place Prefab"
    bl_options = {'UNDO'}
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    target_tree: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        tree = canvas_shortcut_tree(context)
        if tree is None:
            return {'CANCELLED'}
        self.target_tree = tree.name
        return ImportHelper.invoke(self, context, event)

    def execute(self, context):
        tree = bpy.data.node_groups.get(self.target_tree) if self.target_tree else canvas_shortcut_tree(context)
        if not tree:
            return {'CANCELLED'}
        try:
            with open(bpy.path.abspath(self.filepath), encoding='utf-8-sig') as source:
                count = place_canvas_prefab(tree, json.load(source))
        except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Placed {count} prefab elements")
        return {'FINISHED'}


def canvas_event_trait(tree, class_name, package):
    keywords = set("abstract break case cast catch class continue default do dynamic else enum extends extern false final for function if implements import in inline interface macro new null operator overload override package private public return static super switch this throw true try typedef untyped using var while from to".split())
    def identifier(value):
        return bool(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value)) and value not in keywords
    if not identifier(class_name) or not class_name[0].isupper():
        raise ValueError("The Haxe filename must start with an uppercase class name")
    if package and not all(identifier(part) and part[0].islower() for part in package.split('.')):
        raise ValueError("Use lowercase Haxe package segments such as arm or arm.ui")
    events = sorted({node.event_name for node in get_ordered_canvas_nodes(tree) if node.event_name.strip()})
    names = set()
    callbacks = []
    registrations = []
    for event in events:
        stem = re.sub(r'[^a-zA-Z0-9]+', '_', event).strip('_') or 'event'
        name = 'on_' + stem
        suffix = 2
        while name in names:
            name = f'on_{stem}_{suffix}'
            suffix += 1
        names.add(name)
        literal = json.dumps(event, ensure_ascii=False).replace('$', '\\u0024')
        registrations.append(f'            listeners.push(Event.add({literal}, {name}));')
        callbacks.append(f'    function {name}(): Void {{\n        // Handle the canvas event here.\n    }}')
    start = "package" + (' ' + package if package else '') + ';\n\n'
    code = [
        'import armory.system.Event;', 'import armory.system.Event.TEvent;', 'import iron.Trait;', '',
        f'class {class_name} extends Trait {{', '    var listeners: Array<TEvent> = [];', '',
        '    public function new() {', '        super();', '        notifyOnInit(function() {',
        *registrations, '        });', '        notifyOnRemove(function() {',
        '            for (listener in listeners) Event.removeListener(listener);',
        '            listeners = [];', '        });', '    }', '', '\n\n'.join(callbacks), '}', '',
    ]
    return start + '\n'.join(code)


class NODE_OT_armory_canvas_generate_events(Operator, ExportHelper):
    bl_idname = "node.armory_canvas_generate_events"
    bl_label = "Generate Haxe Event Trait"
    filename_ext = ".hx"
    filter_glob: StringProperty(default="*.hx", options={'HIDDEN'})
    package: StringProperty(name="Haxe Package", default="arm")

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "CanvasEvents.hx"
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        tree = _resolve_active_canvas_tree(context)
        if not tree:
            return {'CANCELLED'}
        try:
            class_name = os.path.splitext(os.path.basename(self.filepath))[0]
            _canvas_atomic_write(self.filepath, canvas_event_trait(tree, class_name, self.package.strip()))
        except (ValueError, OSError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Generated {class_name}.hx")
        return {'FINISHED'}


class NODE_OT_armory_canvas_gizmo(Operator):
    bl_idname = "node.armory_canvas_gizmo"
    bl_label = "Interact with Armory Canvas"
    bl_options = {'REGISTER', 'UNDO'}

    mode: StringProperty(default='NONE')

    def invoke(self, context, event):
        space = context.space_data
        if not space or space.type != 'NODE_EDITOR' or space.tree_type != 'ArmoryCanvasTreeType':
            return {'PASS_THROUGH'}

        ntree = space.edit_tree
        if not ntree or not getattr(ntree, "show_canvas_overlay", True):
            return {'PASS_THROUGH'}

        v2d = context.region.view2d
        mouse_r = Vector((event.mouse_region_x, event.mouse_region_y))
        active = ntree.nodes.active
        show_gizmos = ntree.show_gizmos and ntree.as_pointer() not in _canvas_simulations

        draw_order = canvas_paint_order(ntree)
        mouse_v = v2d.region_to_view(event.mouse_region_x, event.mouse_region_y)

        if event.ctrl and event.alt:
            if (not show_gizmos or not isinstance(active, ArmoryCanvasNodeBase)
                    or not active.select or element_selection_locked(active)
                    or not active.is_effectively_visible()):
                return {'PASS_THROUGH'}
            self._start_transform(context, event, active, 'PIVOT', canvas_element_pose(active, ntree)[0])
            self.start_pivot = tuple(active.pivot)
            self.pivot_origin = Vector(canvas_pivot_view(active, ntree))
            self.pivot_targets = [canvas_bounds(get_element_corners_view(active, ntree)[0]),
                                  *canvas_snap_targets(ntree, [active])]
            return {'RUNNING_MODAL'}

        if show_gizmos:
            for group, center in reversed(list(canvas_group_badges(ntree, v2d, draw_order))):
                if max(abs(mouse_r.x - center[0]), abs(mouse_r.y - center[1])) <= 8:
                    if event.value == 'DOUBLE_CLICK':
                        if group.group_locked:
                            set_canvas_group_locked(group, False, True)
                    elif event.shift:
                        set_canvas_group_locked(group, not group.group_locked, True)
                    elif not group.group_locked:
                        set_canvas_group_locked(group, not group.children_locked)
                    context.area.tag_redraw()
                    return {'FINISHED'}

        if event.value == 'DOUBLE_CLICK':
            return {'PASS_THROUGH'}

        selected, scalable, quad = canvas_scale_selection(ntree) if show_gizmos else ((), (), ())
        multi_scale = len(selected) > 1
        if quad:
            corners, edges, circles = canvas_scale_controls(quad, v2d)
            if multi_scale:
                for mode, point in zip((True, False), circles):
                    if (mouse_r - point).length <= 8:
                        ntree.proportional_gizmo_scale = mode
                        context.region.tag_redraw()
                        return {'FINISHED'}
            if scalable:
                handles = [(i, False, point) for i, point in enumerate(edges)
                           if any(canvas_resize_axes(node)[1 - i % 2] for node in scalable)]
                if multi_scale:
                    handles += [(i, True, point) for i, point in enumerate(corners)]
                for index, corner, point in sorted(handles, key=lambda item: (mouse_r - item[2]).length):
                    offset = mouse_r - point
                    if not corner:
                        edge = corners[(index + 1) % 4] - corners[index]
                        direction = edge.normalized()
                        along = offset.dot(direction)
                        offset -= direction * max(-min(7, edge.length * .25),
                                                  min(7, edge.length * .25, along))
                    if offset.length <= (8 if corner else 5):
                        self._start_box_scale(context, event, scalable, quad, index, corner, multi_scale)
                        return {'RUNNING_MODAL'}

        if not event.shift:
            for node in reversed(draw_order):
                if not node.is_effectively_visible() or not canvas_point_in_visible_node(ntree, node, mouse_v):
                    continue
                for axis, track, thumb, travel, limit in canvas_scroll_bars(ntree, node):
                    if travel > 0 and all(lo <= mouse_v[i] <= hi for i, (lo, hi) in enumerate(track)):
                        self.mode = 'SCROLL'
                        self.target_node = node
                        self.scroll_axis = axis
                        self.scroll_start = mouse_v[axis]
                        self.scroll_value = canvas_scroll_offset(node)[axis]
                        self.scroll_original = self.scroll_value
                        self.scroll_factor = limit / travel * (1 if axis == 0 else -1)
                        self.scroll_limit = limit
                        if not thumb[axis][0] <= self.scroll_start <= thumb[axis][1]:
                            center = sum(thumb[axis]) / 2
                            node.preview_scroll[axis] = max(0, min(limit, self.scroll_value + (self.scroll_start - center) * self.scroll_factor))
                            self.scroll_value = node.preview_scroll[axis]
                        context.window_manager.modal_handler_add(self)
                        context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
        if show_gizmos and not event.shift:
            target = parent_gizmo_position(ntree, active, v2d)
            if target is not None and (mouse_r - Vector(target[1])).length <= 8:
                activate_canvas_node(ntree, target[0])
                context.area.tag_redraw()
                return {'FINISHED'}
            width, height = canvas_preview_size(ntree)
            corner = Vector(v2d.view_to_region(
                width, canvas_view_top(ntree) - height, clip=False))
            if (mouse_r - corner).length <= 10:
                self.mode = 'RESIZE_CANVAS'
                self.start_size = (ntree.canvas_width, ntree.canvas_height)
                self.resize_factor = 100.0 / ntree.canvas_resolution_scale
                self.start_mouse_v = Vector(v2d.region_to_view(*mouse_r))
                self.had_view_top = '_canvas_view_top' in ntree
                self.start_dirty = ntree.is_dirty
                context.window_manager.modal_handler_add(self)
                return {'RUNNING_MODAL'}
        if (show_gizmos and not multi_scale and not event.shift and isinstance(active, ArmoryCanvasNodeBase) and active.select
                and not active.group_locked and not element_selection_locked(active)
                and active.is_effectively_visible()
                and canvas_visible_bounds(ntree, active) is not None):
            corners_v, center_v, rot_v = get_element_corners_view(active, ntree)
            rot_r = Vector(v2d.view_to_region(*rot_v, clip=False))
            if (mouse_r - rot_r).length < 14.0:
                self._start_transform(context, event, active, 'ROTATE', center_v)
                return {'RUNNING_MODAL'}

            if active_layout_parent(active) is not None:
                if any(canvas_resize_axes(active)):
                    for idx, corner in enumerate(corners_v):
                        if (mouse_r - Vector(v2d.view_to_region(*corner, clip=False))).length < 14:
                            self._start_box_scale(context, event, [active], tuple(map(Vector, corners_v)), idx, True, False)
                            return {'RUNNING_MODAL'}
            else:
                for idx, corner in enumerate(corners_v):
                    corner_r = Vector(v2d.view_to_region(*corner, clip=False))
                    if (mouse_r - corner_r).length < 14.0:
                        self._start_transform(context, event, active, 'SCALE_CORNER',
                                              center_v, corner_idx=idx, corners_v=corners_v)
                        return {'RUNNING_MODAL'}

        for node in reversed(draw_order):
            if not node.is_effectively_visible():
                continue
            if canvas_point_in_visible_node(ntree, node, mouse_v):
                node = locked_group_for(node) or node
                if element_selection_locked(node):
                    continue
                _corners, center_v, _rotate = get_element_corners_view(node, ntree)
                if event.shift:
                    if node.select:
                        node.select = False
                        if ntree.nodes.active == node:
                            selected = canvas_selected_nodes(ntree)
                            activate_canvas_node(ntree, selected[-1] if selected else None, exclusive=False)
                    else:
                        activate_canvas_node(ntree, node, exclusive=False)
                    context.area.tag_redraw()
                    return {'FINISHED'}
                move_selection = canvas_movable_selection(ntree) if node.select else []
                activate_canvas_node(ntree, node, exclusive=len(move_selection) < 2)

                if layout_child_position(ntree, node) is not None:
                    context.area.tag_redraw()
                    return {'FINISHED'}
                self._start_transform(context, event, node, 'MOVE', center_v, moving=move_selection)
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        if point_in_quad(mouse_v, canvas_box_points(canvas_art_bounds(ntree))):
            return bpy.ops.node.armory_canvas_marquee('INVOKE_DEFAULT')
        return {'PASS_THROUGH'}

    def _start_box_scale(self, context, event, nodes, quad, index, corner, multi):
        tree = nodes[0].id_data
        self.mode = 'SCALE_BOX'
        self.start_dirty = tree.is_dirty
        self.region_key = context.region.as_pointer()
        self.scale_axes = ((quad[1] - quad[0]).normalized(), (quad[3] - quad[0]).normalized())
        if corner:
            self.scale_handle = quad[index]
            self.scale_anchor = quad[(index + 2) % 4]
            self.scale_enabled_axes = (True, True)
        else:
            self.scale_handle = (quad[index] + quad[(index + 1) % 4]) * .5
            self.scale_anchor = (quad[(index + 2) % 4] + quad[(index + 3) % 4]) * .5
            self.scale_enabled_axes = (index % 2 == 1, index % 2 == 0)
        self.scale_start_mouse = Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))
        self.scale_multi = multi
        self.scale_targets = canvas_snap_targets(tree, nodes) if tree.smart_snap else ()
        self.scale_states = {
            node: (canvas_element_pose(node, tree)[0].copy(), node.width, node.height,
                   node.pos_x, node.pos_y, {key: node.get(key) for key in
                   ('width', 'height', '_stretch_width', '_stretch_height')})
            for node in nodes
        }
        context.window_manager.modal_handler_add(self)

    def _update_box_scale(self, context, event, tree, mouse):
        handle = self.scale_handle + mouse - self.scale_start_mouse
        point = tuple((value, value) for value in handle)
        correction, snapped = canvas_alignment_snap(tree, context.region, point, Vector((0, 0)),
                                                     self.scale_targets, bypass=event.ctrl)
        original = self.scale_handle - self.scale_anchor
        current = handle + correction - self.scale_anchor
        factors = [1.0, 1.0]
        for i, axis in enumerate(self.scale_axes):
            if self.scale_enabled_axes[i]:
                span = original.dot(axis)
                size = current.dot(axis) * (1 if span > 0 else -1)
                grid = tree.snap_scale and not event.ctrl and not any(snapped)
                factors[i] = max(1, snap_val(size, tree.grid_size, grid)) / max(1, abs(span))
        proportional = event.shift or (self.scale_multi and tree.proportional_gizmo_scale)
        if not self.scale_multi:
            node = next(iter(self.scale_states))
            proportional |= int(node.armory_type) == 1 and node.keep_aspect_ratio and not node.slice_enabled
        if proportional:
            factor = max((factors[i] for i in range(2) if self.scale_enabled_axes[i]),
                         key=lambda value: abs(value - 1))
            factors = [factor, factor]
        sx, sy = factors
        for node, (center, width, height, _, _, _) in self.scale_states.items():
            relative = center - self.scale_anchor
            target = self.scale_anchor.copy()
            for axis, factor in zip(self.scale_axes, factors):
                target += axis * relative.dot(axis) * factor
            changes = {axis: clamp_canvas_size(node, index, round(size * factor))
                       for index, (axis, size, factor, editable) in enumerate(zip(
                           ('width', 'height'), (width, height), (sx, sy), canvas_resize_axes(node)))
                       if editable and (self.scale_enabled_axes[index] or proportional)}
            set_canvas_transform(node, **changes)
            if active_layout_parent(node) is None:
                canvas_reposition_visual_center(tree, node, target)

    def _update_pivot(self, context, tree, mouse):
        node = self.target_node
        target = self.pivot_origin + mouse - self.start_mouse_v
        point = tuple((value, value) for value in target)
        correction, snapped = canvas_alignment_snap(tree, context.region, point, Vector((0, 0)),
                                                     self.pivot_targets)
        target += correction
        if tree.snap_to_grid:
            target.x = snap_val(target.x, tree.grid_size, not snapped[0])
            top = canvas_view_top(tree)
            target.y = top - snap_val(top - target.y, tree.grid_size, not snapped[1])
        center, angle = canvas_element_pose(node, tree)
        offset = target - center
        local = transform_point(offset.x, offset.y, Vector((0, 0)), math.cos(angle), -math.sin(angle))
        width, height = self.start_size
        pivot = tuple(max(0.0, min(1.0, value)) for value in (.5 + local[0] / width, .5 + local[1] / height))
        if any(abs(old - new) > 1e-6 for old, new in zip(node.pivot, pivot)):
            node.pivot = pivot
            if active_layout_parent(node) is None:
                canvas_reposition_visual_center(tree, node, center)

    def _start_transform(self, context, event, node, mode, center_v, corner_idx=0, corners_v=None, moving=None):
        self.mode = mode
        self.target_node = node
        v2d = context.region.view2d
        self.start_mouse_v = Vector(v2d.region_to_view(event.mouse_region_x, event.mouse_region_y))
        tree = node.id_data
        self.region_key = context.region.as_pointer()
        self.start_dirty = tree.is_dirty
        moving = moving if moving and node in moving else [node]
        self.move_positions = {target: (target.pos_x, target.pos_y) for target in moving}
        self.start_bounds = canvas_union_bounds([canvas_bounds(get_element_corners_view(target, tree)[0])
                                                 for target in moving])
        self.snap_targets = canvas_snap_targets(tree, moving) if mode in {'MOVE', 'SCALE_CORNER'} and tree.smart_snap else ()
        self.start_pos = (node.pos_x, node.pos_y)
        self.start_size = (max(1, node.width), max(1, node.height))
        self.start_stored_size = {key: node.get(key) for key in ('width', 'height', '_stretch_width', '_stretch_height')}
        self.start_rot = node.rotation
        self.center_v = Vector(center_v)
        self.rotation_pivot = Vector(canvas_pivot_view(node, tree))
        self.start_vec = self.start_mouse_v - self.rotation_pivot

        if mode == 'SCALE_CORNER' and corners_v is not None:
            self.corner_idx = corner_idx
            opp_idx = (corner_idx + 2) % 4
            self.anchor_v = Vector(corners_v[opp_idx])
            signs = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
            self.sign_x, self.sign_y = signs[corner_idx]

            _, angle = canvas_element_pose(node, tree)
            self.parent_angle = angle + node.rotation
            wx, wy = node.get_world_pos(tree)
            self.layout_center = Vector((wx + node.width / 2, canvas_view_top(tree) - wy - node.height / 2))
            self.right_axis = Vector((math.cos(angle), math.sin(angle)))
            self.up_axis = Vector((-math.sin(angle), math.cos(angle)))
            self.aspect_ratio = self.start_size[0] / float(self.start_size[1])

        context.window_manager.modal_handler_add(self)

    def modal(self, context, event):
        ntree = context.space_data.edit_tree
        if self.mode in {'SCALE_BOX', 'PIVOT'}:
            if event.type == 'MOUSEMOVE':
                mouse = Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))
                if self.mode == 'SCALE_BOX':
                    self._update_box_scale(context, event, ntree, mouse)
                else:
                    self._update_pivot(context, ntree, mouse)
                context.region.tag_redraw()
            elif event.type in {'ESC', 'RIGHTMOUSE'}:
                if self.mode == 'SCALE_BOX':
                    for node, (_, _, _, x, y, stored) in self.scale_states.items():
                        node.pos_x, node.pos_y = x, y
                        for key, value in stored.items():
                            if value is None:
                                node.pop(key, None)
                            else:
                                node[key] = value
                else:
                    self.target_node.pivot = self.start_pivot
                    self.target_node.pos_x, self.target_node.pos_y = self.start_pos
                invalidate_canvas_geometry(ntree)
                ntree.is_dirty = self.start_dirty
                _canvas_guides.pop(self.region_key, None)
                context.area.tag_redraw()
                return {'CANCELLED'}
            elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                _canvas_guides.pop(self.region_key, None)
                context.area.tag_redraw()
                return {'FINISHED'}
            return {'RUNNING_MODAL'}
        if self.mode == 'SCROLL':
            if event.type in {'ESC', 'RIGHTMOUSE'}:
                self.target_node.preview_scroll[self.scroll_axis] = self.scroll_original
                context.area.tag_redraw()
                return {'CANCELLED'}
            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                return {'FINISHED'}
            if event.type == 'MOUSEMOVE':
                point = context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y)
                amount = self.scroll_value + (point[self.scroll_axis] - self.scroll_start) * self.scroll_factor
                self.target_node.preview_scroll[self.scroll_axis] = max(0, min(self.scroll_limit, amount))
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if self.mode == 'RESIZE_CANVAS':
            if event.type == 'MOUSEMOVE':
                mouse = Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))
                delta = (mouse - self.start_mouse_v) * self.resize_factor
                use_snap = ntree.snap_to_grid and not event.ctrl
                dw, dh = delta.x, -delta.y
                if ntree.canvas_resolution_rotated:
                    dw, dh = dh, dw
                ntree.canvas_width = max(64, snap_val(self.start_size[0] + dw, ntree.grid_size, use_snap))
                ntree.canvas_height = max(64, snap_val(self.start_size[1] + dh, ntree.grid_size, use_snap))
                context.area.tag_redraw()
            elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                return {'FINISHED'}
            elif event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
                ntree.canvas_width, ntree.canvas_height = self.start_size
                if not self.had_view_top:
                    ntree.pop('_canvas_view_top', None)
                ntree.is_dirty = self.start_dirty
                context.area.tag_redraw()
                return {'CANCELLED'}
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE':
            v2d = context.region.view2d
            cur_v = Vector(v2d.region_to_view(event.mouse_region_x, event.mouse_region_y))
            node = self.target_node
            ch = canvas_view_top(ntree)

            use_snap = ntree.snap_to_grid and not event.ctrl
            g_step = ntree.grid_size
            old_guides = _canvas_guides.get(self.region_key, [])
            changed = False

            if self.mode == 'MOVE':
                delta = cur_v - self.start_mouse_v
                delta, snapped = canvas_alignment_snap(
                    ntree, context.region, self.start_bounds, delta, self.snap_targets, bypass=event.ctrl)
                raw_x = self.start_pos[0] + delta.x
                raw_y = self.start_pos[1] - delta.y
                dx = snap_val(raw_x, g_step, use_snap and not snapped[0]) - self.start_pos[0]
                dy = snap_val(raw_y, g_step, use_snap and not snapped[1]) - self.start_pos[1]
                for target, (x, y) in self.move_positions.items():
                    changed |= set_canvas_transform(target, pos_x=x + dx, pos_y=y + dy)

            elif self.mode == 'SCALE_CORNER':
                point_bounds = ((cur_v.x, cur_v.x), (cur_v.y, cur_v.y))
                correction, snapped = canvas_alignment_snap(ntree, context.region, point_bounds,
                    Vector((0, 0)), self.snap_targets, bypass=event.ctrl)
                diag_v = cur_v + correction - self.anchor_v
                proj_w = diag_v.dot(self.right_axis) * self.sign_x
                proj_h = diag_v.dot(self.up_axis) * self.sign_y

                snap_scale = ntree.snap_scale and not event.ctrl
                new_w = max(4, snap_val(proj_w, g_step, snap_scale and not snapped[0]))
                new_h = max(4, snap_val(proj_h, g_step, snap_scale and not snapped[1]))

                image_aspect = int(node.armory_type) == 1 and node.keep_aspect_ratio and not node.slice_enabled
                if event.shift or image_aspect:
                    ratio = self.aspect_ratio
                    if image_aspect:
                        img_db = get_canvas_image(node)
                        if img_db and len(img_db.size) >= 2:
                            iw, ih = img_db.size[0], img_db.size[1]
                            if iw > 0 and ih > 0:
                                ratio = iw / float(ih)
                    if new_w / ratio > new_h:
                        new_h = max(4, int(round(new_w / ratio)))
                    else:
                        new_w = max(4, int(round(new_h * ratio)))

                    ray = self.right_axis * (self.sign_x * ratio) + self.up_axis * self.sign_y
                    unit = self.anchor_v + ray
                    corner_bounds = ((unit.x, unit.x), (unit.y, unit.y))
                    fitted, aligned = canvas_snap_scale_factor(ntree, context.region, corner_bounds,
                        self.anchor_v, new_h, self.snap_targets, bypass=event.ctrl)
                    if aligned:
                        new_h, new_w = max(4, round(fitted)), max(4, round(fitted * ratio))

                new_w = clamp_canvas_size(node, 0, new_w)
                new_h = clamp_canvas_size(node, 1, new_h)
                clamped_corner_v = (
                    self.anchor_v
                    + self.right_axis * (self.sign_x * float(new_w))
                    + self.up_axis * (self.sign_y * float(new_h))
                )
                guides = _canvas_guides.get(self.region_key, ())
                _canvas_guides[self.region_key] = [
                    (i, pos, min(lo, self.anchor_v[1-i]), max(hi, self.anchor_v[1-i]))
                    for i, pos, lo, hi in guides if abs(clamped_corner_v[i] - pos) <= 1
                ]
                new_center_v = (self.anchor_v + clamped_corner_v) * 0.5

                offset = new_center_v - self.center_v
                new_center_v = Vector(transform_point(offset.x, offset.y, self.layout_center,
                    math.cos(self.parent_angle), -math.sin(self.parent_angle)))
                new_vx = new_center_v.x - new_w * 0.5
                new_vy = new_center_v.y - new_h * 0.5
                world_x = int(round(new_vx))
                world_y = int(round(ch - new_vy - new_h))

                changed |= set_canvas_transform(node, width=new_w, height=new_h)
                anc_x, anc_y = node.get_anchor_origin(ntree)
                changed |= set_canvas_transform(node, pos_x=int(world_x - anc_x), pos_y=int(world_y - anc_y))

            elif self.mode == 'ROTATE':
                cur_vec = cur_v - self.rotation_pivot
                if cur_vec.length > 1.0 and self.start_vec.length > 1.0:
                    ang_delta = self.start_vec.angle_signed(cur_vec)
                    changed |= set_canvas_transform(
                        node, rotation=snap_canvas_rotation(ntree, self.start_rot + ang_delta, event.ctrl))

            if changed or old_guides != _canvas_guides.get(self.region_key, []):
                context.region.tag_redraw()
            return {'RUNNING_MODAL'}

        elif event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'RELEASE':
            _canvas_guides.pop(self.region_key, None)
            context.area.tag_redraw()
            return {'FINISHED'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            node = self.target_node
            for target, position in self.move_positions.items():
                target.pos_x, target.pos_y = position
            for key, value in self.start_stored_size.items():
                if value is None:
                    node.pop(key, None)
                else:
                    node[key] = value
            node.rotation = self.start_rot
            invalidate_canvas_geometry(ntree)
            ntree.is_dirty = self.start_dirty
            _canvas_guides.pop(self.region_key, None)
            context.area.tag_redraw()
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

# -------------------------------------------------------------------------
# 6. Hierarchical Outliner List & Asset Selection Operator
# -------------------------------------------------------------------------
class UI_OT_select_asset(Operator, ImportHelper):
    """Opens a filtered file browser showing supported image and font formats."""
    bl_idname = "ui_canvas.select_asset"
    bl_label = "Select UI Asset"
    bl_description = "Browse for a supported image or font asset file"
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(
        default="*.png;*.jpg;*.jpeg;*.webp;*.tga;*.bmp;*.ttf;*.otf;*.woff;*.woff2;*.ttc",
        options={'HIDDEN'},
        maxlen=255,
    )
    use_relative_path: BoolProperty(
        name="Relative Path",
        description="Store path relative to the current blend file",
        default=True,
    )

    button_background: BoolProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        if self.button_background:
            self.filter_glob = ';'.join('*' + suffix for suffix in sorted(IMAGE_EXTENSIONS))
        return ImportHelper.invoke(self, context, event)

    def execute(self, context):
        node = getattr(context, "node", None)
        if not isinstance(node, ArmoryCanvasNodeBase):
            ntree = _resolve_active_canvas_tree(context)
            if ntree and isinstance(ntree.nodes.active, ArmoryCanvasNodeBase):
                node = ntree.nodes.active

        if not node:
            self.report({'WARNING'}, "No active UI node found")
            return {'CANCELLED'}

        abs_path = os.path.abspath(bpy.path.abspath(self.filepath))
        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in ALL_ALLOWED_EXTENSIONS:
            self.report({'ERROR'}, f"Unsupported file extension: {ext}")
            return {'CANCELLED'}

        final_path = self.filepath
        if self.use_relative_path and bpy.data.is_saved:
            try:
                final_path = bpy.path.relpath(abs_path)
            except ValueError:
                final_path = abs_path

        if self.button_background:
            if ext not in IMAGE_EXTENSIONS:
                self.report({'ERROR'}, "Choose an image for the button background")
                return {'CANCELLED'}
            node.button_image_path = final_path
        else:
            node.asset_path = final_path

        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == 'NODE_EDITOR':
                    area.tag_redraw()

        return {'FINISHED'}

_canvas_property_clipboard = []

_CANVAS_COMMON_PROPERTIES = (
    'event_name', 'rotation', 'visible', 'alignment', 'anchor',
    'use_element_color', 'element_color', 'pivot', 'stretch_mode', 'width', 'height',
)
_CANVAS_TEXT_PROPERTIES = ('text_content', 'font_mode', 'font_size', 'asset_path', 'asset_name')
_CANVAS_TYPE_PROPERTIES = {
    0: ('multiline',),
    1: ('asset_path', 'asset_name', 'keep_aspect_ratio'),
    2: ('use_custom_button_states', 'color_text', 'color_hover', 'color_press'),
    6: ('use_custom_button_states', 'color_text', 'color_hover', 'preview_checked'),
    7: ('radio_style', 'preview_radio_selected'),
    8: ('use_custom_button_states', 'color_text', 'color_hover',
        'preview_combo_items', 'preview_combo_index', 'preview_combo_open'),
    9: ('use_custom_button_states', 'color_text', 'color_hover', 'preview_slider_value'),
    10: ('use_custom_button_states', 'color_text', 'color_hover', 'preview_input_text'),
    11: ('color_hover', 'preview_input_text', 'preview_key_hover'),
    20: ('preview_input_text',),
}
for _progress_type in (18, 19):
    _CANVAS_TYPE_PROPERTIES[_progress_type] = (
        'progress_total', 'progress_at', 'use_progress_color', 'color_progress', 'strength',
    )
for _outline_type in (13, 15, 17):
    _CANVAS_TYPE_PROPERTIES[_outline_type] = ('strength',)


def canvas_property_names(node):
    kind = int(node.armory_type)
    text = _CANVAS_TEXT_PROPERTIES if kind in TEXT_CAPABLE_TYPES else ()
    typography = ('use_local_padding', 'text_offset', 'text_vertical_align', 'text_padding_vertical') if kind in TEXT_LAYOUT_TYPES else ()
    colors = ('use_custom_button_states', 'color_text') if kind in {7, 11, 20} else ()
    extras = ['offset_position', 'offset_rotation', 'offset_scale_xy', 'offset_visual_only', 'min_size', 'max_size', 'line_enabled', 'line_color', 'line_thickness', 'line_position']
    if kind == 0:
        extras.extend(('wrap_text', 'wrap_width'))
    if kind in {1, 2}:
        extras.extend(('slice_borders', 'slice_enabled'))
    if kind in CANVAS_DISABLED_TYPES:
        extras.append('disabled')
    if kind in CANVAS_STYLE_TYPES:
        extras.extend(('shadow_enabled', 'shadow_offset', 'shadow_color'))
    if kind in CANVAS_RADIUS_TYPES:
        extras.append('corner_radius')
    if kind in LAYOUT_CONTAINER_TYPES:
        extras.extend(('grid_rows', 'grid_columns', 'layout_use_child_scale', 'scroll_axes', 'scroll_content',
                       'scroll_bar_size', 'scroll_bar_color', 'scroll_track_color'))
    if kind in {2, 6}:
        extras.extend(('control_variant', 'use_end_color', 'end_color', 'color_press'))
        if kind == 2:
            extras.append('button_image_path')
        extras.extend(('step_min', 'step_max', 'step_amount', 'step_initial') if kind == 2 else ('switch_initial',))
    return text + typography + colors + _CANVAS_TYPE_PROPERTIES.get(kind, ()) + _CANVAS_COMMON_PROPERTIES + tuple(extras)


class NODE_OT_armory_canvas_properties_clipboard(Operator):
    bl_idname = "node.armory_canvas_properties_clipboard"
    bl_label = "Canvas Element Properties"
    bl_options = {'UNDO'}

    action: EnumProperty(items=[('COPY', "Copy Properties", ""), ('PASTE', "Paste Properties", "")])

    def execute(self, context):
        tree = context.space_data.node_tree
        if not isinstance(tree, ArmoryCanvasTree):
            return {'CANCELLED'}
        nodes = [node for node, _depth, _last in build_hierarchical_outliner_order(tree)
                 if node.select and not element_selection_locked(node)]
        if not nodes:
            self.report({'WARNING'}, "Select at least one element")
            return {'CANCELLED'}

        if self.action == 'COPY':
            _canvas_property_clipboard.clear()
            for node in nodes:
                props = {}
                for key in canvas_property_names(node):
                    value = getattr(node, key)
                    props[key] = tuple(value) if getattr(node.bl_rna.properties[key], "is_array", False) else value
                _canvas_property_clipboard.append((int(node.armory_type), props))
            self.report({'INFO'}, f"Copied properties from {len(nodes)} element(s)")
            return {'FINISHED'}

        count = len(_canvas_property_clipboard)
        if count not in (1, len(nodes)):
            self.report({'WARNING'}, "Copy one element or select the same number of destination elements")
            return {'CANCELLED'}
        for index, node in enumerate(nodes):
            source_type, props = _canvas_property_clipboard[0 if count == 1 else index]
            dest_type = int(node.armory_type)
            for key in canvas_property_names(node):
                if key not in props:
                    continue
                if key in ('asset_path', 'asset_name') and (source_type == 1) != (dest_type == 1):
                    continue
                setattr(node, key, props[key])
        context.area.tag_redraw()
        self.report({'INFO'}, f"Pasted properties to {len(nodes)} element(s)")
        return {'FINISHED'}


class NODE_OT_armory_canvas_toggle_selection(Operator):
    bl_idname = "node.armory_canvas_toggle_selection"
    bl_label = "Toggle Element Selection"
    bl_description = "Include or exclude this element in the selection without changing the active element"
    bl_options = {'INTERNAL'}

    node_name: StringProperty(options={'SKIP_SAVE'})
    action: EnumProperty(
        items=[('TOGGLE', "Toggle Selection", ""), ('SELECT', "Select All", ""),
               ('DESELECT', "Deselect All", "")],
        default='TOGGLE', options={'SKIP_SAVE'},
    )

    def execute(self, context):
        tree = context.space_data.node_tree
        if not isinstance(tree, ArmoryCanvasTree):
            return {'CANCELLED'}
        if self.action == 'TOGGLE':
            node = tree.nodes.get(self.node_name)
            if not isinstance(node, ArmoryCanvasNodeBase) or element_selection_locked(node):
                return {'CANCELLED'}
            node.select = not node.select
        else:
            for node in get_ordered_canvas_nodes(tree):
                node.select = self.action == 'SELECT' and not element_selection_locked(node)
        context.area.tag_redraw()
        return {'FINISHED'}


class NODE_OT_armory_canvas_toggle_group_lock(Operator):
    bl_idname = "node.armory_canvas_toggle_group_lock"
    bl_label = "Lock or Unlock Element"
    bl_description = "Lock descendants. Shift-click to include the parent"
    include_parent: BoolProperty(options={'SKIP_SAVE','HIDDEN'})

    def invoke(self, context, event):
        self.include_parent = event.shift
        return self.execute(context)
    bl_options = {'UNDO'}

    node_name: StringProperty(options={'SKIP_SAVE'})

    def execute(self, context):
        tree = context.space_data.node_tree
        if not isinstance(tree, ArmoryCanvasTree):
            return {'CANCELLED'}
        node = tree.nodes.get(self.node_name)
        if not isinstance(node, ArmoryCanvasNodeBase) or not set_canvas_group_locked(node, not (node.group_locked if self.include_parent or not node_has_children(node) else node.children_locked), self.include_parent):
            return {'CANCELLED'}
        context.area.tag_redraw()
        return {'FINISHED'}


class NODE_UL_armory_canvas_elements(UIList):
    bl_idname = "NODE_UL_armory_canvas_elements"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if not isinstance(item, ArmoryCanvasNodeBase):
            return

        hierarchy = canvas_hierarchy_cache(data)
        depth = hierarchy['depths'].get(item, 0)
        has_kids = node_has_children(item)

        row = layout.row(align=True)

        for _ in range(depth):
            indent_sub = row.row(align=True)
            indent_sub.scale_x = 0.45
            indent_sub.separator()

        if has_kids:
            exp_icon = 'DISCLOSURE_TRI_DOWN' if item.outliner_expanded else 'DISCLOSURE_TRI_RIGHT'
            row.prop(item, "outliner_expanded", text="", icon=exp_icon, emboss=False)
        else:
            spacer_sub = row.row(align=True)
            spacer_sub.scale_x = 0.85
            spacer_sub.label(text="", icon='BLANK1')

        parent_of = hierarchy['parents'].get
        locked_ancestor = locked_group_for(item, parent_of)
        row.label(text=item.element_name)

        lock_row = row.row(align=True)
        lock_row.enabled = locked_ancestor is None
        locked = item.children_locked or item.group_locked or locked_ancestor is not None
        icon = 'LOCKED' if locked else 'UNLOCKED'
        lock_row.operator("node.armory_canvas_toggle_group_lock", text="", icon=icon,
                          depress=locked, emboss=False).node_name = item.name

        visibility = row.row(align=True)
        visibility.enabled = locked_ancestor is None and not item.group_locked
        eye_icon = 'HIDE_OFF' if item.is_effectively_visible(parent_of) else 'HIDE_ON'
        visibility.prop(item, "visible", text="", icon=eye_icon, emboss=False)

    def filter_items(self, context, data, propname):
        nodes = list(getattr(data, propname))
        hierarchy = canvas_hierarchy_cache(data)
        node_to_rank = hierarchy['ranks']
        parents = hierarchy['parents']
        collapsed = {}
        filter_name = self.filter_name.lower()
        for node, _, _ in hierarchy['hierarchy']:
            parent = parents.get(node)
            collapsed[node] = parent in collapsed and (collapsed[parent] or not parent.outliner_expanded)

        flt_flags = []
        flt_neworder = []
        next_rank = len(node_to_rank)

        for node in nodes:
            if isinstance(node, ArmoryCanvasNodeBase):
                if collapsed.get(node, False):
                    flt_flags.append(0)
                elif filter_name and filter_name not in node.element_name.lower():
                    flt_flags.append(0)
                else:
                    flt_flags.append(self.bitflag_filter_item)
            else:
                flt_flags.append(0)

            if node in node_to_rank:
                flt_neworder.append(node_to_rank[node])
            else:
                flt_neworder.append(next_rank)
                next_rank += 1

        return flt_flags, flt_neworder

def fit_canvas_stretch(node):
    if active_layout_parent(node) is not None:
        return
    axes = STRETCH_VALUES[node.stretch_mode]
    for bit, axis, offset in ((1, 'width', 'pos_x'), (2, 'height', 'pos_y')):
        if axes & bit:
            node['_stretch_' + axis] = 0
            setattr(node, offset, 0)
    mark_tree_dirty(node, None)


class NODE_OT_armory_canvas_fit_stretch(Operator):
    bl_idname = "node.armory_canvas_fit_stretch"
    bl_label = "Reset Stretch Insets"
    bl_description = "Clear the position and size offsets on stretched axes"
    bl_options = {'UNDO'}

    def execute(self, context):
        node = getattr(context, "canvas_element", None) or context.space_data.edit_tree.nodes.active
        if not isinstance(node, ArmoryCanvasNodeBase) or node.stretch_mode == 'NONE' or element_selection_locked(node):
            return {'CANCELLED'}
        fit_canvas_stretch(node)
        context.area.tag_redraw()
        return {'FINISHED'}


def canvas_apply_anchor_preset(node, column, row, set_pivot=False, set_position=False):
    tree = node.id_data
    position = node.get_world_pos(tree)
    stretch = ('NONE', 'WIDTH', 'HEIGHT', 'BOTH')[(column == 3) + 2 * (row == 3)]
    column, row = (1 if column == 3 else column), (1 if row == 3 else row)
    node.anchor = str(row * 3 + column)
    node.stretch_mode = stretch
    if set_pivot:
        node.pivot = (column / 2, 1 - row / 2)
    if set_position:
        node.pos_x = node.pos_y = 0
        if stretch != 'NONE':
            fit_canvas_stretch(node)
    else:
        x, y = node.get_anchor_origin(tree)
        node.pos_x, node.pos_y = position[0] - x, position[1] - y


class NODE_OT_armory_canvas_anchor_preset(Operator):
    bl_idname = "node.armory_canvas_anchor_preset"
    bl_label = "Apply Anchor Preset"
    bl_description = "Set anchors and stretch. Shift also sets pivot. Alt also aligns position"
    bl_options = {'UNDO'}

    column: IntProperty(min=0, max=3)
    row: IntProperty(min=0, max=3)
    set_pivot: BoolProperty(options={'SKIP_SAVE'})
    set_position: BoolProperty(options={'SKIP_SAVE'})

    def invoke(self, context, event):
        self.set_pivot, self.set_position = event.shift, event.alt
        return self.execute(context)

    def execute(self, context):
        tree = canvas_shortcut_tree(context)
        node = getattr(context, "canvas_element", None) or (tree.nodes.active if tree else None)
        if (not isinstance(node, ArmoryCanvasNodeBase) or node.id_data != tree
                or element_selection_locked(node) or active_layout_parent(node) is not None):
            return {'CANCELLED'}
        canvas_apply_anchor_preset(node, self.column, self.row, self.set_pivot, self.set_position)
        context.area.tag_redraw()
        return {'FINISHED'}


class NODE_PT_armory_canvas_anchor_presets(Panel):
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_label = "Anchor Presets"
    bl_ui_units_x = 20

    @classmethod
    def poll(cls, context):
        tree = canvas_shortcut_tree(context)
        return tree is not None and isinstance(tree.nodes.active, ArmoryCanvasNodeBase)

    def draw(self, context):
        tree = canvas_shortcut_tree(context)
        node = getattr(context, "canvas_element", None) or tree.nodes.active
        layout = self.layout
        layout.context_pointer_set("canvas_element", node)
        layout.operator_context = 'INVOKE_DEFAULT'
        layout.enabled = not element_selection_locked(node)
        grid = layout.column(align=True)
        grid.enabled = active_layout_parent(node) is None
        grid.label(text="Shift to set pivot · Alt to align position")
        matrix = grid.grid_flow(row_major=True, columns=5, even_columns=True,
                                even_rows=True, align=True)
        for label in ("", "Left", "Center", "Right", "Stretch"):
            cell = matrix.row()
            cell.alignment = 'CENTER'
            cell.label(text=label)
        for y, label in enumerate(("Top", "Middle", "Bottom", "Stretch")):
            matrix.label(text=label)
            for x in range(4):
                mode = ('NONE', 'WIDTH', 'HEIGHT', 'BOTH')[(x == 3) + 2 * (y == 3)]
                anchor = (1 if y == 3 else y) * 3 + (1 if x == 3 else x)
                glyph = {'NONE': '•', 'WIDTH': '↔', 'HEIGHT': '↕', 'BOTH': '⤢'}[mode]
                cell = matrix.row()
                cell.scale_y = 1.25
                op = cell.operator("node.armory_canvas_anchor_preset", text=glyph,
                    depress=node.anchor == str(anchor) and node.stretch_mode == mode)
                op.column, op.row = x, y
        grid.separator()
        grid.prop(node, "anchor")
        grid.prop(node, "stretch_mode")
        grid.operator_menu_enum("node.armory_canvas_align_element", "placement", text="Align Position")
        draw_canvas_pair(layout, node, "pivot", "Pivot", ("X", "Y"))


class NODE_OT_armory_canvas_align_element(Operator):
    bl_idname = "node.armory_canvas_align_element"
    bl_label = "Align Element"
    bl_description = "Place the element in its parent or canvas without changing its anchor"
    bl_options = {'UNDO'}

    placement: EnumProperty(name="Align", items=CANVAS_ANCHOR_ITEMS + [
        ('WIDTH', "SWidth", "Fill the parent or canvas width and keep following it"),
        ('HEIGHT', "SHeight", "Fill the parent or canvas height and keep following it"),
        ('BOTH', "SBoth", "Fill both dimensions of the parent or canvas"),
    ])

    def execute(self, context):
        tree = context.space_data.node_tree
        if not isinstance(tree, ArmoryCanvasTree):
            return {'CANCELLED'}
        node = getattr(context, "canvas_element", None) or tree.nodes.active
        if not isinstance(node, ArmoryCanvasNodeBase) or node.id_data != tree or element_selection_locked(node):
            return {'CANCELLED'}
        if self.placement in STRETCH_VALUES:
            node.stretch_mode = self.placement
            fit_canvas_stretch(node)
            context.area.tag_redraw()
            return {'FINISHED'}
        anchor_x, anchor_y = node.get_anchor_origin(tree)
        x, y = node.get_anchor_origin(tree, anchor=self.placement)
        node.pos_x, node.pos_y = x - anchor_x, y - anchor_y
        context.area.tag_redraw()
        return {'FINISHED'}


class NODE_OT_armory_canvas_move_element(Operator):
    bl_idname = "node.armory_canvas_move_element"
    bl_label = "Move Canvas Element"
    bl_description = "Reorder elements within their parent hierarchy"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(
        items=[('UP', "Up", ""), ('DOWN', "Down", ""),
               ('BACK', "Send to Back", ""), ('FRONT', "Bring to Front", "")],
        default='UP'
    )

    @classmethod
    def poll(cls, context):
        return canvas_shortcut_tree(context) is not None

    def execute(self, context):
        ntree = canvas_shortcut_tree(context)
        if ntree is None:
            return {'CANCELLED'}

        target_node = ntree.nodes.active
        if not isinstance(target_node, ArmoryCanvasNodeBase):
            if 0 <= ntree.active_element_index < len(ntree.nodes):
                cand = ntree.nodes[ntree.active_element_index]
                if isinstance(cand, ArmoryCanvasNodeBase):
                    target_node = cand
        if not isinstance(target_node, ArmoryCanvasNodeBase):
            return {'CANCELLED'}

        parent = get_parent_node(target_node)
        ordered = sorted(get_ordered_canvas_nodes(ntree), key=lambda n: (n.z_order, n.name))
        siblings = [node for node in ordered if get_parent_node(node) == parent]

        if target_node not in siblings:
            return {'CANCELLED'}

        sib_idx = siblings.index(target_node)
        target_sib_idx = {
            'UP': sib_idx - 1, 'DOWN': sib_idx + 1,
            'FRONT': 0, 'BACK': len(siblings) - 1,
        }[self.direction]

        if target_sib_idx == sib_idx or not (0 <= target_sib_idx < len(siblings)):
            return {'CANCELLED'}

        swap_node = siblings[target_sib_idx]
        first, second = ordered.index(target_node), ordered.index(swap_node)
        if self.direction in {'FRONT', 'BACK'}:
            ordered.insert(second, ordered.pop(first))
        else:
            ordered[first], ordered[second] = ordered[second], ordered[first]
        for rank, node in enumerate(ordered):
            if node.z_order != rank:
                node.z_order = rank
        ntree.is_dirty = True
        sync_active_element_index(ntree, target_node)

        if context.area:
            context.area.tag_redraw()
        return {'FINISHED'}

def canvas_shortcut_tree(context):
    space = getattr(context, "space_data", None)
    tree = getattr(space, "edit_tree", None)
    if tree is None:
        tree = getattr(space, "node_tree", None)
    return tree if isinstance(tree, ArmoryCanvasTree) else None


def snap_canvas_rotation(tree, angle, bypass=False):
    if not tree.snap_rotation or bypass:
        return angle
    step = math.radians(tree.rotation_snap_degrees)
    return round(angle / step) * step


class NODE_OT_armory_canvas_element_popup(Operator):
    bl_idname = "node.armory_canvas_element_popup"
    bl_label = "Element Properties"

    @classmethod
    def poll(cls, context):
        tree = canvas_shortcut_tree(context)
        return tree is not None and isinstance(tree.nodes.active, ArmoryCanvasNodeBase)

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=380)

    def draw(self, context):
        tree = canvas_shortcut_tree(context)
        if tree is not None and isinstance(tree.nodes.active, ArmoryCanvasNodeBase):
            tree.nodes.active.draw_element_properties(context, self.layout)

    def execute(self, context):
        return {'FINISHED'}


class NODE_MT_armory_canvas_parent(Menu):
    bl_idname = "NODE_MT_armory_canvas_parent"
    bl_label = "Parent Elements"

    def draw(self, context):
        self.layout.operator("node.armory_canvas_parent_selected", text="Parent Objects").keep_transform = False
        self.layout.operator("node.armory_canvas_parent_selected", text="Parent (Keep Transform)").keep_transform = True
        self.layout.operator("node.armory_canvas_unparent", text="Unparent").all_selected = True


class NODE_OT_armory_canvas_parent_menu(Operator):
    bl_idname = "node.armory_canvas_parent_menu"
    bl_label = "Parent Elements"

    def invoke(self, context, event):
        tree = canvas_shortcut_tree(context)
        if tree is None or len(canvas_selected_nodes(tree)) < 2:
            return {'PASS_THROUGH'}
        return bpy.ops.wm.call_menu(name=NODE_MT_armory_canvas_parent.bl_idname)


class NODE_OT_armory_canvas_duplicate_hierarchy(Operator):
    bl_idname = "node.armory_canvas_duplicate_hierarchy"
    bl_label = "Duplicate Element Hierarchy"
    bl_options = {'UNDO'}

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):
        tree = canvas_shortcut_tree(context)
        if tree is None:
            return {'PASS_THROUGH'}
        selected = canvas_selected_nodes(tree)
        if not selected:
            return {'PASS_THROUGH'}

        active = tree.nodes.active
        children = {}
        ordered = get_ordered_canvas_nodes(tree)
        for node in ordered:
            children.setdefault(get_parent_node(node), []).append(node)
        included = set()
        pending = selected.copy()
        while pending:
            node = pending.pop()
            if node not in included:
                included.add(node)
                pending.extend(children.get(node, ()))

        copies = {}
        next_z_order = max((node.z_order for node in ordered), default=-1) + 1
        for index, node in enumerate(n for n in ordered if n in included):
            duplicate = tree.nodes.new(node.bl_idname)
            for prop in ArmoryCanvasNodeBase.__annotations__:
                if prop not in {'stretch_mode', 'width', 'height', 'z_order'}:
                    value = getattr(node, prop)
                    rna = node.bl_rna.properties[prop]
                    setattr(duplicate, prop, tuple(value) if getattr(rna, 'is_array', False) else value)
            for key in ('width', 'height', '_stretch_mode', '_stretch_width', '_stretch_height',
                        '_armory_canvas_original_element', '_armory_imported_asset_path'):
                if key in node:
                    duplicate[key] = node[key]
            duplicate.location = node.location + Vector((36, -36))
            duplicate.z_order = next_z_order + index
            copies[node] = duplicate

        # Rebuild links before restoring responsive sizes, since reparenting can
        # recalculate stretch offsets.
        for node, duplicate in copies.items():
            parent = get_parent_node(node)
            if parent is not None:
                set_node_parent(tree, duplicate, copies.get(parent, parent))
            for key in ('width', 'height', '_stretch_width', '_stretch_height'):
                if key in node:
                    duplicate[key] = node[key]
                elif key in duplicate:
                    del duplicate[key]
            if parent not in copies and active_layout_parent(duplicate) is None:
                duplicate.pos_x += 20
                duplicate.pos_y += 20

        normalize_z_orders(tree)
        activate_canvas_node(tree, copies.get(active, copies[selected[-1]]))
        for node in selected:
            if not element_selection_locked(copies[node]):
                copies[node].select = True
        context.area.tag_redraw()
        return {'FINISHED'}


class NODE_OT_armory_canvas_transform_selected(Operator):
    bl_idname = "node.armory_canvas_transform_selected"
    bl_label = "Transform Canvas Elements"
    bl_options = {'UNDO'}

    operation: EnumProperty(items=[('MOVE', "Move", ""), ('ROTATE', "Rotate", ""), ('SCALE', "Scale", "")])

    def invoke(self, context, event):
        tree = canvas_shortcut_tree(context)
        if tree is None or not tree.show_canvas_overlay:
            return {'PASS_THROUGH'}
        selection = [node for node in canvas_selected_nodes(tree) if node.is_effectively_visible()]
        self.states = {}
        for node in selection:
            if any(is_ancestor_of(parent, node) for parent in selection if parent != node):
                continue
            if self.operation == 'MOVE' and active_layout_parent(node) is not None:
                continue
            if self.operation == 'SCALE' and not any(canvas_resize_axes(node)):
                continue
            wx, wy = node.get_world_pos(tree)
            self.states[node] = (node.pos_x, node.pos_y, node.rotation, node.width, node.height,
                                 wx, wy, {key: node.get(key) for key in
                                          ('width', 'height', '_stretch_width', '_stretch_height')})
        if not self.states:
            return {'PASS_THROUGH'}
        self.start_dirty = tree.is_dirty
        self.axis = None
        self.region_key = context.region.as_pointer()
        self.start_bounds = canvas_union_bounds([canvas_bounds(get_element_corners_view(node, tree)[0])
                                                 for node in self.states])
        self.snap_targets = canvas_snap_targets(tree, self.states) if self.operation in {'MOVE', 'SCALE'} and tree.smart_snap else ()
        self.start = Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))
        self.pivot = sum((Vector((wx + w / 2, canvas_view_top(tree) - wy - h / 2))
                          for _, _, _, w, h, wx, wy, _ in self.states.values()), Vector((0, 0))) / len(self.states)
        if self.operation == 'SCALE' and len(self.states) > 1:
            self.pivot = Vector(tuple((low + high) * .5 for low, high in self.start_bounds))
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        tree = canvas_shortcut_tree(context)
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            for node, (px, py, rotation, _, _, _, _, stored) in self.states.items():
                node.pos_x, node.pos_y, node.rotation = px, py, rotation
                for key, value in stored.items():
                    if value is not None:
                        node[key] = value
                    elif key in node:
                        del node[key]
            if tree is not None:
                invalidate_canvas_geometry(tree)
                tree.is_dirty = self.start_dirty
            _canvas_guides.pop(self.region_key, None)
            context.area.tag_redraw()
            return {'CANCELLED'}
        if event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            _canvas_guides.pop(self.region_key, None)
            context.area.tag_redraw()
            return {'FINISHED'}
        if event.type in {'X', 'Y'} and event.value == 'PRESS':
            self.axis = None if self.axis == event.type else event.type
        elif event.type != 'MOUSEMOVE':
            return {'RUNNING_MODAL'}

        mouse = Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))
        delta = mouse - self.start
        if self.axis == 'X':
            delta.y = 0
        elif self.axis == 'Y':
            delta.x = 0
        start_vec = self.start - self.pivot
        current_vec = start_vec + delta
        angle = start_vec.angle_signed(current_vec, 0.0) if start_vec.length > 1 else -delta.x * .01
        factor = current_vec.length / start_vec.length if start_vec.length > 1 else 1 + (delta.x + delta.y) * .01
        if self.axis:
            index = 0 if self.axis == 'X' else 1
            if self.operation == 'SCALE':
                component = start_vec[index]
                factor = 1 + delta[index] / (component if abs(component) >= 1 else 100)
            elif self.operation == 'ROTATE':
                angle = -delta[index] * .01
        factor = max(.01, factor)
        scale_snapped = False
        old_guides = _canvas_guides.get(self.region_key, [])
        changed = False
        if self.operation == 'SCALE':
            factor, scale_snapped = canvas_snap_scale_factor(tree, context.region, self.start_bounds,
                self.pivot, factor, self.snap_targets, self.axis, event.ctrl)
        sx, sy = (1 if self.axis == 'Y' else factor), (1 if self.axis == 'X' else factor)
        if self.operation == 'ROTATE':
            start_rotation = next(iter(self.states.values()))[2] if len(self.states) == 1 else 0.0
            angle = snap_canvas_rotation(tree, start_rotation + angle, event.ctrl) - start_rotation
        if self.operation == 'MOVE':
            delta, snapped = canvas_alignment_snap(tree, context.region, self.start_bounds, delta,
                                                   self.snap_targets, axis=self.axis, bypass=event.ctrl)
            snap = tree.snap_to_grid and not event.ctrl
            first_x, first_y = next(iter(self.states.values()))[:2]
            dx = snap_val(first_x + delta.x, tree.grid_size, snap and not snapped[0]) - first_x
            dy = snap_val(first_y - delta.y, tree.grid_size, snap and not snapped[1]) - first_y
            for node, (px, py, *_) in self.states.items():
                changed |= set_canvas_transform(node, pos_x=px + (0 if self.axis == 'Y' else dx),
                                                pos_y=py + (0 if self.axis == 'X' else dy))
        else:
            for node, (px, py, rotation, w, h, wx, wy, _) in self.states.items():
                if self.operation == 'ROTATE':
                    changed |= set_canvas_transform(node, rotation=rotation + angle)
                    if len(self.states) > 1 and active_layout_parent(node) is None:
                        cx, cy = wx + w / 2, wy + h / 2
                        pivot_y = canvas_view_top(tree) - self.pivot.y
                        dx, dy = cx - self.pivot.x, cy - pivot_y
                        cos_a, sin_a = math.cos(angle), math.sin(angle)
                        ax, ay = node.get_anchor_origin(tree)
                        changed |= set_canvas_transform(
                            node, pos_x=round(self.pivot.x + dx * cos_a - dy * sin_a - w / 2 - ax),
                            pos_y=round(pivot_y + dx * sin_a + dy * cos_a - h / 2 - ay))
                    continue
                snap_scale = tree.snap_scale and not event.ctrl and not scale_snapped
                width = max(1, snap_val(w * sx, tree.grid_size, snap_scale)) if self.axis != 'Y' else w
                height = max(1, snap_val(h * sy, tree.grid_size, snap_scale)) if self.axis != 'X' else h
                editable_width, editable_height = canvas_resize_axes(node)
                if editable_width:
                    changed |= set_canvas_transform(node, width=width)
                if editable_height:
                    changed |= set_canvas_transform(node, height=height)
                if active_layout_parent(node) is None:
                    cx, cy = wx + w / 2, wy + h / 2
                    pivot_y = canvas_view_top(tree) - self.pivot.y
                    ax, ay = node.get_anchor_origin(tree)
                    changed |= set_canvas_transform(
                        node, pos_x=round(self.pivot.x + (cx - self.pivot.x) * sx - node.width / 2 - ax),
                        pos_y=round(pivot_y + (cy - pivot_y) * sy - node.height / 2 - ay))
        if changed or old_guides != _canvas_guides.get(self.region_key, []):
            context.region.tag_redraw()
        return {'RUNNING_MODAL'}


class NODE_OT_armory_canvas_parent_selected(Operator):
    bl_idname = "node.armory_canvas_parent_selected"
    bl_label = "Parent Selected to Active"
    bl_description = "Make the active element the parent of the other selected elements"
    bl_options = {'REGISTER', 'UNDO'}

    keep_transform: BoolProperty(name="Keep Visual Position", default=True)

    def execute(self, context):
        ntree = canvas_shortcut_tree(context)
        if ntree is None:
            return {'CANCELLED'}

        active = ntree.nodes.active
        if not isinstance(active, ArmoryCanvasNodeBase):
            self.report({'WARNING'}, "Select a target parent node first")
            return {'CANCELLED'}

        selected_children = [
            n for n in ntree.nodes
            if isinstance(n, ArmoryCanvasNodeBase) and n.select and n != active and not element_selection_locked(n)
        ]
        if not selected_children:
            self.report({'WARNING'}, "Select one or more child nodes")
            return {'CANCELLED'}

        if any(is_ancestor_of(child, active) for child in selected_children):
            self.report({'WARNING'}, "Deselect ancestors of the active element before parenting")
            return {'CANCELLED'}

        positions = {child: (child.get_world_pos(ntree), child.width, child.height)
                     for child in selected_children} if self.keep_transform else {}
        for child in selected_children:
            if get_parent_node(child) == active:
                continue
            set_node_parent(ntree, child, active)
            if self.keep_transform:
                (old_wx, old_wy), old_w, old_h = positions[child]
                if active_layout_parent(child) is not None:
                    child.ignore_parent_layout = True
                child.width, child.height = old_w, old_h
                anc_x, anc_y = child.get_anchor_origin(ntree)
                child.pos_x = int(old_wx - anc_x)
                child.pos_y = int(old_wy - anc_y)

        active.outliner_expanded = True
        ntree.is_dirty = True
        sync_active_element_index(ntree, active)
        context.area.tag_redraw()
        self.report({'INFO'}, f"Parented {len(selected_children)} element(s) to {active.element_name}")
        return {'FINISHED'}

class NODE_OT_armory_canvas_unparent(Operator):
    bl_idname = "node.armory_canvas_unparent"
    bl_label = "Clear Parent"
    bl_description = "Detach active element from its parent container"
    bl_options = {'REGISTER', 'UNDO'}

    all_selected: BoolProperty(default=False, options={'SKIP_SAVE'})

    def execute(self, context):
        ntree = context.space_data.edit_tree
        if not isinstance(ntree, ArmoryCanvasTree):
            return {'CANCELLED'}
        active = ntree.nodes.active
        candidates = canvas_selected_nodes(ntree) if self.all_selected else [active]
        states = [(node, node.get_world_pos(ntree), node.width, node.height) for node in candidates
                  if isinstance(node, ArmoryCanvasNodeBase) and get_parent_node(node) is not None]
        if not states:
            return {'CANCELLED'}
        for node, (x, y), width, height in states:
            set_node_parent(ntree, node, None)
            node.width, node.height = width, height
            ax, ay = node.get_anchor_origin(ntree)
            node.pos_x, node.pos_y = x - ax, y - ay
        ntree.is_dirty = True
        context.area.tag_redraw()
        return {'FINISHED'}


class NODE_OT_armory_canvas_add_element(Operator):
    bl_idname = "node.armory_canvas_add_element"
    bl_label = "Add Canvas Element"
    bl_description = "Add an Armory Canvas UI element node"
    bl_options = {'REGISTER', 'UNDO'}

    node_type: StringProperty(default='ArmoryCanvasFilledRectNode')
    preset: EnumProperty(items=[('DEFAULT', "Default", ""), ('STEPPER', "Numeric Stepper", ""),
                                ('SWITCH', "Toggle Switch", ""), ('SCROLL', "Scroll View", "")],
                         default='DEFAULT', options={'HIDDEN'})

    def execute(self, context):
        ntree = _resolve_active_canvas_tree(context)
        if not ntree:
            self.report({'ERROR'}, "No active Armory Canvas tree")
            return {'CANCELLED'}

        if self.preset == 'STEPPER':
            node = ntree.nodes.new(type='ArmoryCanvasButtonNode')
            node.element_name = "Numeric Stepper"
            node.control_variant = 'STEPPER'
        elif self.preset == 'SWITCH':
            node = ntree.nodes.new(type='ArmoryCanvasCheckNode')
            node.element_name = "Toggle Switch"
            node.control_variant = 'SWITCH'
        elif self.preset == 'SCROLL':
            node = ntree.nodes.new(type='ArmoryCanvasEmptyNode')
            node.element_name = "Scroll View"
            node.scroll_axes = (False, True)
            node.width, node.height = 320, 240
        else:
            node = ntree.nodes.new(type=self.node_type)
        count = len([n for n in ntree.nodes if isinstance(n, ArmoryCanvasNodeBase)])
        node.location = (float(ntree.canvas_width) + 120.0, canvas_view_top(ntree) - (count - 1) * 60.0)

        if context.area:
            context.area.tag_redraw()
        return {'FINISHED'}

# -------------------------------------------------------------------------
# 7. Import / Export Operators & Panels
# -------------------------------------------------------------------------
def _resolve_active_canvas_tree(context):
    space = getattr(context, "space_data", None)
    if space and getattr(space, "node_tree", None):
        if space.node_tree.bl_idname == 'ArmoryCanvasTreeType':
            return space.node_tree
    for ng in bpy.data.node_groups:
        if ng.bl_idname == 'ArmoryCanvasTreeType':
            return ng
    return None

class NODE_OT_import_armory_canvas(Operator):
    bl_idname = "node.import_armory_canvas"
    bl_label = "Load"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH", default="")
    filter_glob: StringProperty(default="*.json;*.files", options={'HIDDEN'})

    def execute(self, context):
        tree = _resolve_active_canvas_tree(context)
        if not tree:
            tree = bpy.data.node_groups.new(name="ArmoryCanvas", type='ArmoryCanvasTreeType')
            if context.space_data and context.space_data.type == 'NODE_EDITOR':
                context.space_data.node_tree = tree

        if not self.filepath:
            return {'CANCELLED'}

        abs_path = os.path.abspath(bpy.path.abspath(self.filepath))
        if abs_path.endswith(".files"):
            json_cand = os.path.splitext(abs_path)[0]
            if os.path.isfile(json_cand):
                abs_path = json_cand

        count = import_canvas_json_to_tree(tree, abs_path)
        tree.export_filepath = abs_path if not bpy.data.filepath else bpy.path.relpath(abs_path)
        if context.area:
            context.area.tag_redraw()
        self.report({'INFO'}, f"Loaded {count} elements and assets.")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class NODE_OT_export_armory_canvas(Operator):
    bl_idname = "node.export_armory_canvas"
    bl_label = "Export"
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype="FILE_PATH", default="")
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    force_dialog: BoolProperty(default=False, options={'SKIP_SAVE'})
    force_delay: BoolProperty(default=False, options={'SKIP_SAVE'})

    def execute(self, context):
        tree = _resolve_active_canvas_tree(context)
        if not tree:
            return {'CANCELLED'}

        raw_path = self.filepath or tree.export_filepath
        out_path = export_canvas_tree_to_json(tree, raw_path)
        tree.export_filepath = out_path if not bpy.data.filepath else bpy.path.relpath(out_path)
        if context.area:
            context.area.tag_redraw()
        self.report({'INFO'}, f"Saved canvas and .files archive to {out_path}")
        return {'FINISHED'}

    def invoke(self, context, event):
        tree = _resolve_active_canvas_tree(context)
        if not tree:
            return {'CANCELLED'}
        if tree.export_filepath and not self.force_dialog:
            self.filepath = tree.export_filepath
            return self.execute(context)
        self.filepath = f"{tree.canvas_name or tree.name}.json"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

ELEMENT_GROUPS = [
    ("Basic", "ARMORY_MT_add_basic", 'OBJECT_DATA', [ArmoryCanvasEmptyNode, ArmoryCanvasTextNode, ArmoryCanvasImageNode]),
    ("Buttons", "ARMORY_MT_add_buttons", 'CHECKBOX_HLT', [ArmoryCanvasButtonNode, ArmoryCanvasCheckNode, ArmoryCanvasRadioNode]),
    ("Inputs", "ARMORY_MT_add_inputs", 'GREASEPENCIL', [ArmoryCanvasTextInputNode, ArmoryCanvasTextAreaNode, ArmoryCanvasKeyInputNode, ArmoryCanvasComboBoxNode, ArmoryCanvasSliderNode]),
    ("Shapes", "ARMORY_MT_add_shapes", 'MESH_PLANE', [ArmoryCanvasRectNode, ArmoryCanvasFilledRectNode, ArmoryCanvasCircleNode, ArmoryCanvasFilledCircleNode, ArmoryCanvasTriangleNode, ArmoryCanvasFilledTriangleNode]),
    ("Progress Bars", "ARMORY_MT_add_progress", 'STATUSBAR', [ArmoryCanvasProgressNode, ArmoryCanvasCProgressNode]),
]

class ARMORY_MT_add_basic(Menu):
    bl_idname = "ARMORY_MT_add_basic"
    bl_label = "Basic"
    def draw(self, context):
        for cls in ELEMENT_GROUPS[0][3]:
            self.layout.operator("node.add_node", text=cls.bl_label).type = cls.bl_idname
        self.layout.operator("node.armory_canvas_add_element", text="Scroll View").preset = 'SCROLL'

class ARMORY_MT_add_buttons(Menu):
    bl_idname = "ARMORY_MT_add_buttons"
    bl_label = "Buttons"
    def draw(self, context):
        for cls in ELEMENT_GROUPS[1][3]:
            self.layout.operator("node.add_node", text=cls.bl_label).type = cls.bl_idname
        self.layout.separator()
        self.layout.operator("node.armory_canvas_add_element", text="Numeric Stepper").preset = 'STEPPER'
        self.layout.operator("node.armory_canvas_add_element", text="Toggle Switch").preset = 'SWITCH'

class ARMORY_MT_add_inputs(Menu):
    bl_idname = "ARMORY_MT_add_inputs"
    bl_label = "Inputs"
    def draw(self, context):
        for cls in ELEMENT_GROUPS[2][3]:
            self.layout.operator("node.add_node", text=cls.bl_label).type = cls.bl_idname

class ARMORY_MT_add_shapes(Menu):
    bl_idname = "ARMORY_MT_add_shapes"
    bl_label = "Shapes"
    def draw(self, context):
        for cls in ELEMENT_GROUPS[3][3]:
            self.layout.operator("node.add_node", text=cls.bl_label).type = cls.bl_idname

class ARMORY_MT_add_progress(Menu):
    bl_idname = "ARMORY_MT_add_progress"
    bl_label = "Progress Bars"
    def draw(self, context):
        for cls in ELEMENT_GROUPS[4][3]:
            self.layout.operator("node.add_node", text=cls.bl_label).type = cls.bl_idname

def draw_builtin_subpanel(layout, idname, label, icon='NONE', default_closed=False):
    header, body = layout.panel(idname, default_closed=default_closed)
    if icon != 'NONE':
        header.label(text=label, icon=icon)
    else:
        header.label(text=label)
    return body

class ArmoryCanvasSidebarPanelBase(Panel):
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Armory Canvas'

    @classmethod
    def poll(cls, context):
        space = getattr(context, "space_data", None)
        return bool(space and space.tree_type == 'ArmoryCanvasTreeType')

    @staticmethod
    def get_tree_or_warn(context, layout):
        ntree = getattr(context.space_data, "node_tree", None)
        if not ntree:
            layout.label(text="Create or select an Armory Canvas tree", icon='INFO')
            return None
        return ntree

class NODE_PT_armory_canvas_add_elements(ArmoryCanvasSidebarPanelBase):
    bl_label = "Add elements"
    bl_idname = "NODE_PT_armory_canvas_add_elements"
    bl_order = 0
    bl_options = {'DEFAULT_CLOSED'}
    def draw(self, context):
        layout = self.layout
        ntree = self.get_tree_or_warn(context, layout)
        if not ntree:
            return
        for grp_label, menu_idname, grp_icon, grp_classes in ELEMENT_GROUPS:
            sub_body = draw_builtin_subpanel(layout, f"{menu_idname}_sub", grp_label, icon=grp_icon, default_closed=False)
            if sub_body:
                col = sub_body.column(align=True)
                for cls in grp_classes:
                    op = col.operator("node.armory_canvas_add_element", text=cls.bl_label)
                    op.node_type = cls.bl_idname
                if menu_idname == 'ARMORY_MT_add_basic':
                    col.operator("node.armory_canvas_add_element", text="Scroll View").preset = 'SCROLL'
                elif menu_idname == 'ARMORY_MT_add_buttons':
                    col.operator("node.armory_canvas_add_element", text="Numeric Stepper").preset = 'STEPPER'
                    col.operator("node.armory_canvas_add_element", text="Toggle Switch").preset = 'SWITCH'

class NODE_OT_armory_canvas_apply_size(Operator):
    bl_idname = "node.armory_canvas_apply_size"
    bl_label = "Apply Canvas Size"
    bl_options = {'UNDO'}

    width: IntProperty(default=1024, min=64)
    height: IntProperty(default=1024, min=64)

    def execute(self, context):
        tree = context.space_data.edit_tree
        if not isinstance(tree, ArmoryCanvasTree):
            return {'CANCELLED'}
        tree.canvas_width, tree.canvas_height = self.width, self.height
        context.area.tag_redraw()
        return {'FINISHED'}


class NODE_OT_armory_canvas_custom_size(Operator):
    bl_idname = "node.armory_canvas_custom_size"
    bl_label = "Custom Canvas Size"
    bl_options = {'UNDO'}

    width: IntProperty(name="Width", min=64)
    height: IntProperty(name="Height", min=64)

    def invoke(self, context, event):
        tree = context.space_data.edit_tree
        if not isinstance(tree, ArmoryCanvasTree):
            return {'CANCELLED'}
        self.width, self.height = tree.canvas_width, tree.canvas_height
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        tree = context.space_data.edit_tree
        if not isinstance(tree, ArmoryCanvasTree):
            return {'CANCELLED'}
        tree.canvas_width, tree.canvas_height = self.width, self.height
        context.area.tag_redraw()
        return {'FINISHED'}


class NODE_OT_armory_canvas_save_size(Operator):
    bl_idname = "node.armory_canvas_save_size"
    bl_label = "Save Canvas Size Preset"
    bl_options = {'UNDO'}

    preset_name: StringProperty(name="Name")

    def invoke(self, context, event):
        tree = context.space_data.edit_tree
        if not isinstance(tree, ArmoryCanvasTree):
            return {'CANCELLED'}
        self.preset_name = f"{tree.canvas_width} × {tree.canvas_height}"
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        tree = context.space_data.edit_tree
        if not isinstance(tree, ArmoryCanvasTree):
            return {'CANCELLED'}
        name = self.preset_name.strip()
        if not name or name.casefold() in {item[0].casefold() for item in CANVAS_SIZE_PRESETS}:
            self.report({'ERROR'}, "Enter a unique name for the preset")
            return {'CANCELLED'}
        if any(preset.name.casefold() == name.casefold() for preset in tree.canvas_size_presets):
            self.report({'ERROR'}, "A preset with that name already exists")
            return {'CANCELLED'}
        preset = tree.canvas_size_presets.add()
        preset.name = name
        preset.width, preset.height = tree.canvas_width, tree.canvas_height
        context.area.tag_redraw()
        return {'FINISHED'}


class NODE_OT_armory_canvas_open_preset_folder(Operator):
    bl_idname = "node.armory_canvas_open_preset_folder"
    bl_label = "Open Presets Folder"

    def execute(self, context):
        try:
            path = canvas_preset_file(create=True)
            try:
                with open(path, 'x', encoding='utf-8') as stream:
                    stream.write("[]\n")
            except FileExistsError:
                pass
        except OSError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return bpy.ops.wm.path_open(filepath=os.path.dirname(path))


def reset_canvas_preset_page(operator, context):
    operator.page = 1


class NODE_OT_armory_canvas_size_picker(Operator):
    bl_idname = "node.armory_canvas_size_picker"
    bl_label = "Size Presets"

    category: EnumProperty(
        name="Device",
        items=[(category, category, "") for category in CANVAS_PRESET_CATEGORIES],
        update=reset_canvas_preset_page,
    )
    search: StringProperty(name="Search", options={'TEXTEDIT_UPDATE'}, update=reset_canvas_preset_page)
    page: IntProperty(name="Page", default=1, min=1)

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=780)

    def execute(self, context):
        return {'FINISHED'}

    def draw(self, context):
        tree = context.space_data.edit_tree
        if not isinstance(tree, ArmoryCanvasTree):
            return
        layout = self.layout
        local = [(preset.name, preset.width, preset.height) for preset in tree.canvas_size_presets]
        reserved = {name.casefold() for name, _, _ in (*CANVAS_SIZE_PRESETS, *local)}
        external, invalid = read_external_canvas_presets(reserved)
        presets = [
            (name, width, height, canvas_preset_category(width, height))
            for name, width, height in (*CANVAS_SIZE_PRESETS, *local) if width and height
        ]
        presets.extend(external)

        layout.prop(self, "category", expand=True)
        layout.prop(self, "search", icon='VIEWZOOM')
        query = self.search.strip().casefold()
        entries = [(name, width, height) for name, width, height, category in presets
                   if category == self.category and query in f"{name} {width} × {height}".casefold()]
        column_count = 2 if self.category == 'Mobile' else 1
        rows_per_column = 10
        page_size = column_count * rows_per_column
        page_count = max(1, (len(entries) + page_size - 1) // page_size)
        page = min(self.page, page_count)
        if page_count > 1:
            row = layout.row(align=True)
            row.prop(self, "page")
            row.label(text=f"{page_count} pages")
        first = (page - 1) * page_size
        visible = entries[first:first + page_size]
        if not visible:
            layout.label(text="No matching resolutions")

        columns = layout.row()
        for i in range(column_count):
            column = columns.column(align=True)
            for name, width, height in visible[i * rows_per_column:(i + 1) * rows_per_column]:
                item = column.operator("node.armory_canvas_apply_size", text=f"{name} · {width} × {height}")
                item.width, item.height = width, height

        layout.separator()
        row = layout.row(align=True)
        row.operator("node.armory_canvas_custom_size", text="Custom")
        row.operator("node.armory_canvas_open_preset_folder", icon='FILE_FOLDER')
        if invalid:
            layout.label(text="Check resolution_presets.json for errors or duplicate names", icon='ERROR')


class NODE_PT_armory_canvas_settings(ArmoryCanvasSidebarPanelBase):
    bl_label = "Canvas"
    bl_idname = "NODE_PT_armory_canvas_settings"
    bl_order = 2
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return ArmoryCanvasSidebarPanelBase.poll(context) and bool(getattr(context.space_data, "node_tree", None))

    def draw(self, context):
        ntree = context.space_data.node_tree
        layout = self.layout
        layout.prop(ntree, "canvas_name")
        layout.prop(ntree, "theme_name")
        row = layout.row(align=True)
        row.prop(ntree, "canvas_width")
        row.prop(ntree, "canvas_height")
        row.prop(ntree, "canvas_resolution_rotated", text="", icon='FILE_REFRESH', toggle=True)
        row = layout.row(align=True)
        row.prop(ntree, "canvas_resolution_scale", slider=True)
        row.prop(ntree, "apply_resolution_scale_to_armory", text="Armory", toggle=True)
        width, height = canvas_preview_size(ntree)
        layout.label(text=f"Viewport Resolution  {width} × {height}")
        row = layout.row(align=True)
        row.operator_context = 'INVOKE_DEFAULT'
        row.operator("node.armory_canvas_size_picker", text="Size Presets")
        row.operator("node.armory_canvas_save_size", text="Save Preset", icon='ADD')
        col = layout.column(align=True)
        for label, prop in (("Area", "canvas_bg_color"), ("Border", "canvas_border_color")):
            row = col.split(factor=0.45, align=True)
            row.label(text=f"{label}:")
            row.prop(ntree, prop, text="")
        layout.prop(ntree, "canvas_visible")

class NODE_OT_armory_theme_action(Operator):
    bl_idname = 'node.armory_canvas_theme_action'
    bl_label = 'Manage Zui Theme'
    bl_options = {'REGISTER', 'UNDO'}
    action: EnumProperty(items=[(a,a,'') for a in ('LOAD','SAVE','NEW','COPY','DELETE','PREV','NEXT','APPLY','RENAME')])

    def execute(self, context):
        tree = context.space_data.node_tree
        if not isinstance(tree, ArmoryCanvasTree):
            return {'CANCELLED'}
        path = canvas_theme_path(tree)
        try:
            if self.action == 'LOAD':
                if not theme_import_file(tree, path):
                    self.report({'WARNING'}, 'No _themes.json found: ' + path)
                    return {'CANCELLED'}
            elif self.action == 'SAVE':
                os.makedirs(os.path.dirname(path), exist_ok=True)
                _canvas_atomic_write(path, json.dumps(canvas_theme_list(tree), indent=2, ensure_ascii=False) + '\n')
                self.report({'INFO'}, 'Saved ' + path)
            else:
                themes = canvas_theme_list(tree)
                index = next((i for i,t in enumerate(themes) if t.get('NAME') == tree.theme_name), 0)
                if self.action in {'PREV','NEXT'}:
                    index = (index + (-1 if self.action == 'PREV' else 1)) % len(themes)
                elif self.action in {'NEW','COPY'}:
                    base = dict(DEFAULT_LIGHT_THEME[0]) if self.action == 'NEW' else dict(themes[index])
                    stem = 'New Theme' if self.action == 'NEW' else str(base.get('NAME','Theme')) + ' Copy'
                    name = stem
                    n = 1
                    while any(t.get('NAME') == name for t in themes):
                        name = f'{stem}.{n:03d}'
                        n += 1
                    base['NAME'] = name
                    themes.append(base)
                    index = len(themes)-1
                elif self.action == 'DELETE':
                    if len(themes) <= 1:
                        self.report({'WARNING'}, 'Keep at least one theme')
                        return {'CANCELLED'}
                    themes.pop(index)
                    index = min(index, len(themes)-1)
                elif self.action == 'RENAME':
                    name = tree.theme_edit_name.strip()
                    if not name or any(i != index and t.get('NAME') == name for i,t in enumerate(themes)):
                        self.report({'WARNING'}, 'Theme name empty or already exists')
                        return {'CANCELLED'}
                    themes[index]['NAME'] = name
                tree.theme_json = json.dumps(themes, ensure_ascii=False)
                tree.theme_name = str(themes[index].get('NAME', 'Default Light'))
                theme_load_fields(tree)
            tree.is_dirty = True
            return {'FINISHED'}
        except (OSError, ValueError, TypeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

class NODE_PT_armory_canvas_themes(ArmoryCanvasSidebarPanelBase):
    bl_label = 'Theme'
    bl_idname = 'NODE_PT_armory_canvas_themes'
    bl_order = 1
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        tree = context.space_data.node_tree
        if not isinstance(tree, ArmoryCanvasTree):
            return
        row = layout.row(align=True)
        row.operator('node.armory_canvas_theme_action', text='', icon='TRIA_LEFT').action='PREV'
        row.label(text=tree.theme_name)
        row.operator('node.armory_canvas_theme_action', text='', icon='TRIA_RIGHT').action='NEXT'
        row = layout.row(align=True)
        for action, label in [('NEW','Add'),('COPY','Copy'),('DELETE','Delete')]:
            row.operator('node.armory_canvas_theme_action', text=label).action=action
        row = layout.row(align=True)
        row.prop(tree, 'theme_edit_name', text='Name')
        row.operator('node.armory_canvas_theme_action', text='', icon='CHECKMARK').action='RENAME'
        row = layout.row(align=True)
        row.operator('node.armory_canvas_theme_action', text='Load').action='LOAD'
        row.operator('node.armory_canvas_theme_action', text='Save').action='SAVE'
        layout.label(text='Text')
        layout.prop(tree, 'theme_font_size', text='FONT_SIZE')
        layout.prop(tree, 'theme_text_offset', text='TEXT_OFFSET')
        layout.prop(tree, 'theme_text_col', text='TEXT_COL')
        layout.label(text='Elements')
        for key in ('BUTTON_COL','BUTTON_TEXT_COL','BUTTON_HOVER_COL','ACCENT_COL','ACCENT_HOVER_COL','ACCENT_SELECT_COL'):
            layout.prop(tree, 'theme_'+key.lower(), text=key)
        layout.label(text='Other')
        for key in ('PANEL_BG_COL','WINDOW_BG_COL','LABEL_COL','SEPARATOR_COL','HIGHLIGHT_COL'):
            layout.prop(tree, 'theme_'+key.lower(), text=key)

class NODE_OT_armory_canvas_bulk_locks(Operator):
    bl_idname = "node.armory_canvas_bulk_locks"
    bl_label = "Element Locks"
    bl_options = {'UNDO'}

    action: EnumProperty(items=[('LOCK', "Lock All", ""), ('UNLOCK', "Unlock All", ""),
                                ('INVERT', "Invert All Locks", "")])

    def execute(self, context):
        tree = context.space_data.node_tree
        if not isinstance(tree, ArmoryCanvasTree):
            return {'CANCELLED'}
        for node in get_ordered_canvas_nodes(tree):
            node.group_locked = not node.group_locked if self.action == 'INVERT' else self.action == 'LOCK'
            node.children_locked = node.group_locked and node_has_children(node)
        sync_locked_group_selection(tree)
        context.area.tag_redraw()
        return {'FINISHED'}


class NODE_OT_armory_canvas_delete_groups(Operator):
    bl_idname = "node.armory_canvas_delete_groups"
    bl_label = "Delete Group Parents"
    bl_description = "Remove matching parent elements and preserve surviving children and their canvas positions"
    bl_options = {'UNDO'}

    include_locked: BoolProperty(default=False)

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        tree = context.space_data.node_tree
        if not isinstance(tree, ArmoryCanvasTree):
            return {'CANCELLED'}
        nodes = canvas_paint_order(tree)
        removed = {node for node in nodes if node_has_children(node) and (
            self.include_locked or (not node.group_locked and not locked_group_for(node)))}
        if not removed:
            self.report({'INFO'}, "No matching group parents")
            return {'CANCELLED'}

        reparent = []
        for node in nodes:
            parent = get_parent_node(node)
            if node in removed or parent not in removed:
                continue
            while parent in removed:
                parent = get_parent_node(parent)
            reparent.append((node, parent, node.get_world_pos(tree)))

        # Parents that survive retain their positions before their children are adjusted.
        for node, parent, (x, y) in reparent:
            set_node_parent(tree, node, parent)
            ax, ay = node.get_anchor_origin(tree)
            node.pos_x, node.pos_y = int(x - ax), int(y - ay)
        for node in reversed(nodes):
            if node in removed:
                tree.nodes.remove(node)

        sync_locked_group_selection(tree)
        mark_tree_dirty(tree, context)
        context.area.tag_redraw()
        self.report({'INFO'}, f"Deleted {len(removed)} group parent(s)")
        return {'FINISHED'}


class NODE_MT_armory_canvas_outliner_actions(Menu):
    bl_idname = "NODE_MT_armory_canvas_outliner_actions"
    bl_label = "Outliner Actions"

    def draw(self, context):
        layout = self.layout
        layout.operator("node.armory_canvas_toggle_selection", text="Select All",
                        icon='CHECKBOX_HLT').action = 'SELECT'
        layout.operator("node.armory_canvas_toggle_selection", text="Deselect All",
                        icon='CHECKBOX_DEHLT').action = 'DESELECT'
        layout.separator()
        layout.operator("node.armory_canvas_bulk_locks", text="Lock All", icon='LOCKED').action = 'LOCK'
        layout.operator("node.armory_canvas_bulk_locks", text="Unlock All", icon='UNLOCKED').action = 'UNLOCK'
        layout.operator("node.armory_canvas_bulk_locks", text="Invert All Locks").action = 'INVERT'
        layout.separator()
        layout.operator_context = 'INVOKE_DEFAULT'
        layout.operator("node.armory_canvas_delete_groups", text="Delete All Unlocked Groups",
                        icon='TRASH').include_locked = False
        layout.operator("node.armory_canvas_delete_groups", text="Delete All Groups",
                        icon='TRASH').include_locked = True


class NODE_PT_armory_canvas_outliner_hierarchy(ArmoryCanvasSidebarPanelBase):
    bl_label = "Hierarchy"
    bl_idname = "NODE_PT_armory_canvas_outliner_hierarchy"
    bl_order = 3

    @classmethod
    def poll(cls, context):
        return ArmoryCanvasSidebarPanelBase.poll(context) and bool(getattr(context.space_data, "node_tree", None))

    def draw(self, context):
        layout = self.layout
        ntree = context.space_data.node_tree
        row = layout.row()
        row.template_list("NODE_UL_armory_canvas_elements", "", ntree, "nodes", ntree, "active_element_index", rows=6)
        col_btns = row.column(align=True)
        col_btns.operator("node.armory_canvas_move_element", text="", icon='TRIA_UP').direction = 'UP'
        col_btns.operator("node.armory_canvas_move_element", text="", icon='TRIA_DOWN').direction = 'DOWN'

        col_btns.menu("NODE_MT_armory_canvas_outliner_actions", text="", icon='DOWNARROW_HLT')

        actions = layout.row(align=True)
        actions.operator("node.armory_canvas_properties_clipboard", text="Copy Properties", icon='COPYDOWN').action = 'COPY'
        paste = actions.row(align=True)
        paste.enabled = bool(_canvas_property_clipboard)
        paste.operator("node.armory_canvas_properties_clipboard", text="Paste Properties", icon='PASTEDOWN').action = 'PASTE'

        active = ntree.nodes.active
        if isinstance(active, ArmoryCanvasNodeBase) and not element_selection_locked(active):
            layout.prop(active, "element_name")
            parent = get_parent_node(active)
            parent_name = parent.element_name if parent else "Canvas Root"
            layout.row(align=True).label(text=f"Parent: {parent_name}", icon='LINKED' if parent else 'NODE')
            row_p = layout.row(align=True)
            row_p.prop(ntree, "parent_selector", text="Set Parent")
            if parent:
                row_p.operator("node.armory_canvas_unparent", text="", icon='UNLINKED')
            parenting = layout.row()
            parenting.enabled = any(node.select and node != active
                                    for node in get_ordered_canvas_nodes(ntree))
            parenting.operator("node.armory_canvas_parent_selected", icon='LINKED')

class NODE_PT_armory_canvas_selection(ArmoryCanvasSidebarPanelBase):
    bl_label = "Prefab"
    bl_idname = "NODE_PT_armory_canvas_selection"
    bl_parent_id = "NODE_PT_armory_canvas_outliner_hierarchy"

    def draw(self, context):
        tree = self.get_tree_or_warn(context, self.layout)
        if not tree:
            return
        row = self.layout.row(align=True)
        row.operator("node.armory_canvas_save_prefab", icon='EXPORT')
        row.operator("node.armory_canvas_place_prefab", icon='IMPORT')


class NODE_PT_armory_canvas_element_properties(ArmoryCanvasSidebarPanelBase):
    bl_label = "Element Properties"
    bl_idname = "NODE_PT_armory_canvas_element_properties"
    bl_parent_id = "NODE_PT_armory_canvas_outliner_hierarchy"

    @classmethod
    def poll(cls, context):
        if not super().poll(context):
            return False
        tree = context.space_data.edit_tree
        return tree is not None and isinstance(tree.nodes.active, ArmoryCanvasNodeBase)

    def draw(self, context):
        node = context.space_data.edit_tree.nodes.active
        node.draw_element_properties(context, self.layout)


class NODE_PT_armory_canvas_viewport_and_grid(ArmoryCanvasSidebarPanelBase):
    bl_label = "View"
    bl_idname = "NODE_PT_armory_canvas_viewport_and_grid"
    bl_order = 4
    bl_options = {'DEFAULT_CLOSED'}
    def draw(self, context):
        self.get_tree_or_warn(context, self.layout)

class NODE_PT_armory_canvas_viewport_gizmos_nodes(ArmoryCanvasSidebarPanelBase):
    bl_label = "Settings"
    bl_idname = "NODE_PT_armory_canvas_viewport_gizmos_nodes"
    bl_parent_id = "NODE_PT_armory_canvas_viewport_and_grid"

    @classmethod
    def poll(cls, context):
        return ArmoryCanvasSidebarPanelBase.poll(context) and bool(getattr(context.space_data, "node_tree", None))

    def draw(self, context):
        ntree = context.space_data.node_tree
        layout = self.layout
        draw_canvas_view_settings(layout, ntree)

class NODE_PT_armory_canvas_grid_and_snapping(ArmoryCanvasSidebarPanelBase):
    bl_label = "Grid & Snapping"
    bl_idname = "NODE_PT_armory_canvas_grid_and_snapping"
    bl_parent_id = "NODE_PT_armory_canvas_viewport_and_grid"

    @classmethod
    def poll(cls, context):
        return ArmoryCanvasSidebarPanelBase.poll(context) and bool(getattr(context.space_data, "node_tree", None))

    def draw(self, context):
        ntree = context.space_data.node_tree
        layout = self.layout
        draw_canvas_grid_settings(layout, ntree)


def draw_canvas_grid_settings(layout, ntree):
    row = layout.row(align=True)
    row.prop(ntree, "show_grid", text="Show", toggle=True, icon='GRID')
    row.prop(ntree, "extend_grid", text="Full View", toggle=True, icon='FULLSCREEN_ENTER')
    layout.prop(ntree, "grid_color", text="Grid Color")
    draw_canvas_snapping(layout, ntree)

_ARMORY_RUNTIME_MEMBERS = r'''
	// Extended editor metadata. Older canvases continue through the original renderer.
	static var canvasPlusLayout: Map<Int, CanvasPlusBox>;
	static var canvasPlusOverflow: Map<Int, Array<Float>>;
	static var canvasPlusState: CanvasPlusState;
	static var canvasPlusClip: CanvasPlusRect;
	static var canvasPlusWheelUsed = false;
	static var canvasPlusRenderScale = 1.0;

	static inline function canvasPlusFeatures(element: TElement): Dynamic
		return Reflect.field(element, "_editor_features");

	static inline function canvasPlusField(features: Dynamic, key: String): Dynamic
		return features == null ? null : Reflect.field(features, key);

	static function canvasPlusNumber(features: Dynamic, key: String, fallback: Float): Float {
		var value = canvasPlusField(features, key);
		return value == null ? fallback : value;
	}

	static function canvasPlusBound(element: TElement, axis: Int, size: Int): Int {
		var data = canvasPlusFeatures(element);
		var minimum: Array<Int> = canvasPlusField(data, "min_size");
		var maximum: Array<Int> = canvasPlusField(data, "max_size");
		var low = minimum == null ? 1 : Std.int(Math.max(1, minimum[axis]));
		var high = maximum == null || maximum[axis] == 0 ? 2147483647
			: Std.int(Math.max(low, maximum[axis]));
		return Std.int(Math.max(low, Math.min(high, size)));
	}

	static function canvasPlusRound(value: Float): Int {
		var floor = Math.floor(value);
		return value - floor == .5 ? floor + (floor & 1) : Math.round(value);
	}

	static function canvasPlusStretch(element: TElement, axis: Int, inGroup: Bool): Bool {
		var data = Reflect.field(element, inGroup ? "_editor_layout_stretch" : "_editor_stretch");
		var mode = canvasPlusField(data, "mode");
		return mode == "BOTH" || mode == (axis == 0 ? "WIDTH" : "HEIGHT");
	}

	static function canvasPlusSize(element: TElement, parentSize: Int, axis: Int): Int {
		var size = axis == 0 ? element.width : element.height;
		if (canvasPlusStretch(element, axis, false)) {
			var offsets: Array<Int> = canvasPlusField(Reflect.field(element, "_editor_stretch"), "offsets");
			if (offsets != null && offsets.length == 2) size = parentSize + offsets[axis];
		}
		return canvasPlusBound(element, axis, size);
	}

	static function canvasPlusBegin(ui: Zui, canvas: TCanvas, stock: Bool) {
		canvasPlusRenderScale = ui.SCALE();
		canvasPlusClip = null;
		canvasPlusWheelUsed = false;
		canvasPlusState = cast Reflect.field(canvas, "_canvasPlusState");
		canvasPlusLayout = null;
		canvasPlusOverflow = null;
		var reflow = false;
		for (element in canvas.elements) {
			var group = Reflect.field(element, "_editor_layout_group");
			var mode = canvasPlusField(group, "mode");
			var features = canvasPlusFeatures(element);
			if (features != null && canvasPlusState == null) {
				canvasPlusState = {scroll: new Map(), steps: new Map(), switches: new Map(),
					dragId: null, dragAxis: -1, dragGrab: 0};
				Reflect.setField(canvas, "_canvasPlusState", canvasPlusState);
			}
			if (mode == "GRID" || mode == "VERTICAL" || mode == "HORIZONTAL"
				|| canvasPlusField(features, "min_size") != null || canvasPlusField(features, "max_size") != null
				|| canvasPlusField(features, "scroll_axes") != null) reflow = true;
		}
		if (canvasPlusState != null && !ui.inputDown && !ui.inputReleased) canvasPlusState.dragId = null;
		if (!reflow) return;
		canvasPlusLayout = new Map();
		canvasPlusOverflow = new Map();
		var byId = new Map<Int, TElement>();
		for (element in canvas.elements) byId.set(element.id, element);
		var width = Std.int(canvas.width / Math.max(.001, canvasPlusRenderScale));
		var height = Std.int(canvas.height / Math.max(.001, canvasPlusRenderScale));
		for (element in canvas.elements) {
			if (element.parent == null)
				canvasPlusArrange(element, canvasPlusManual(element, width, height), byId, 0);
		}
	}

	static function canvasPlusManual(element: TElement, parentW: Int, parentH: Int): CanvasPlusBox {
		var manual: Array<Int> = Reflect.field(element, "_editor_layout_manual");
		var width = canvasPlusSize(element, parentW, 0);
		var height = canvasPlusSize(element, parentH, 1);
		var anchor: Int = element.anchor == null ? 0 : element.anchor;
		var x = element.x, y = element.y;
		if (manual != null && manual.length == 3) {
			x = manual[0];
			y = manual[1];
			anchor = manual[2];
		}
		return {x: x, y: y, width: width, height: height, managed: false,
			ax: canvasPlusRound((parentW - width) * (anchor % 3) / 2),
			ay: canvasPlusRound((parentH - height) * Std.int(anchor / 3) / 2)};
	}

	static function canvasPlusArrange(element: TElement, box: CanvasPlusBox,
			byId: Map<Int, TElement>, depth: Int): Array<Float> {
		canvasPlusLayout.set(element.id, box);
		var content: Array<Int> = canvasPlusField(canvasPlusFeatures(element), "scroll_content");
		var axesEnabled: Array<Bool> = canvasPlusField(canvasPlusFeatures(element), "scroll_axes");
		var layoutBox = box;
		if (content != null && content.length == 2 && axesEnabled != null && axesEnabled.length == 2) {
			layoutBox = cast Reflect.copy(box);
			if (axesEnabled[0]) layoutBox.width = Std.int(Math.max(box.width, content[0]));
			if (axesEnabled[1]) layoutBox.height = Std.int(Math.max(box.height, content[1]));
		}
		var far: Array<Float> = [layoutBox.width, layoutBox.height];
		if (element.children == null || depth >= 128) return far;
		var children = new Array<TElement>();
		var group = Reflect.field(element, "_editor_layout_group");
		var mode = canvasPlusField(group, "mode");
		if (mode == "GRID" || mode == "HORIZONTAL" || mode == "VERTICAL") {
			for (index in 0...element.children.length) {
				var child = byId.get(element.children[element.children.length - index - 1]);
				if (child != null && child.visible != false
					&& canvasPlusField(Reflect.field(child, "_editor_layout_group"), "ignore") != true) children.push(child);
			}
			if (canvasPlusField(group, "reverse") == true) children.reverse();
		}
		var boxes = canvasPlusGroup(group, mode, layoutBox, children);
		for (id in element.children) {
			var child = byId.get(id);
			if (child == null || child.visible == false || canvasPlusLayout.exists(id)) continue;
			var childBox = boxes.get(id);
			if (childBox == null) childBox = canvasPlusManual(child, box.width, box.height);
			var childFar = canvasPlusArrange(child, childBox, byId, depth + 1);
			if (canvasPlusField(canvasPlusFeatures(child), "scroll_axes") != null)
				childFar = [childBox.width, childBox.height];
			far[0] = Math.max(far[0], childBox.x + childBox.ax + childFar[0]);
			far[1] = Math.max(far[1], childBox.y + childBox.ay + childFar[1]);
		}
		var axes: Array<Bool> = canvasPlusField(canvasPlusFeatures(element), "scroll_axes");
		if (axes != null && axes.length == 2) {
			canvasPlusOverflow.set(element.id, [
				axes[0] ? Math.max(0, far[0] - box.width) : 0,
				axes[1] ? Math.max(0, far[1] - box.height) : 0,
			]);
		}
		return far;
	}

	static function canvasPlusEqualParts(available: Int, spacing: Int, count: Int): Array<Int> {
		var gap = count > 1 ? Std.int(Math.min(spacing, Math.max(0, Math.floor((available - count) / (count - 1))))) : 0;
		var usable = Std.int(Math.max(count, available - gap * (count - 1)));
		return [Std.int(usable / count), usable % count, gap];
	}

	static function canvasPlusGroup(group: Dynamic, mode: String, parent: CanvasPlusBox,
			children: Array<TElement>): Map<Int, CanvasPlusBox> {
		var boxes = new Map<Int, CanvasPlusBox>();
		var count = children.length;
		if (count == 0) return boxes;
		var padding: Array<Int> = canvasPlusField(group, "padding");
		if (padding == null || padding.length != 4) padding = [0, 0, 0, 0];
		var origin = [padding[0], padding[2]];
		var available = [Std.int(Math.max(0, parent.width - padding[0] - padding[1])),
			Std.int(Math.max(0, parent.height - padding[2] - padding[3]))];
		var spacing = Std.int(canvasPlusNumber(group, "spacing", 0));
		var alignment = Std.parseInt(Std.string(canvasPlusField(group, "alignment")));
		if (alignment == null || alignment < 0 || alignment > 8) alignment = 0;
		var align = [alignment % 3 / 2, Std.int(alignment / 3) / 2];
		var horizontal = mode == "HORIZONTAL";
		var main = horizontal ? 0 : 1;
		var cross = 1 - main;

		var control: Array<Bool> = canvasPlusField(group, "control_size");
		var force: Array<Bool> = canvasPlusField(group, "force_expand");
		if (control == null || control.length != 2) control = [false, false];
		if (force == null || force.length != 2) force = [false, false];
		if (mode == "GRID") {
			var columns = Std.int(Math.max(1, canvasPlusNumber(group, "grid_columns", 2)));
			var rows = Std.int(Math.max(canvasPlusNumber(group, "grid_rows", 2), Math.ceil(count / columns)));
			var parts = [canvasPlusEqualParts(available[0], spacing, columns),
				canvasPlusEqualParts(available[1], spacing, rows)];
			for (i in 0...count) {
				var indices = [i % columns, Std.int(i / columns)];
				var pos = origin.copy();
				var size = available.copy();
				for (axis in 0...2) {
					var part = parts[axis];
					var index = indices[axis];
					var slot = part[0] + (index < part[1] ? 1 : 0);
					size[axis] = canvasPlusBound(children[i], axis, slot);
					pos[axis] += index * (part[0] + part[2]) + Std.int(Math.min(index, part[1]))
						+ canvasPlusRound((slot - size[axis]) * align[axis]);
				}
				boxes.set(children[i].id, canvasPlusCell(pos, size));
			}
			return boxes;
		}
		if (canvasPlusField(group, "proportional") == true) {
			var sizes = [for (child in children) canvasPlusPreferredSize(child)];
			var minimumSlot = 1;
			for (i in 0...count) {
				var minimum = control[main] ? canvasPlusBound(children[i], main, 1) : sizes[i][main];
				minimumSlot = Std.int(Math.max(minimumSlot, minimum));
			}
			var usable = Std.int(Math.max(available[main] - spacing * (count - 1), minimumSlot * count));
			var share = Std.int(usable / count), remainder = usable % count;
			var cursor = 0;
			for (i in 0...count) {
				var slots = available.copy();
				slots[main] = share + (i < remainder ? 1 : 0);
				var pos = origin.copy();
				pos[main] += cursor;
				var size = sizes[i];
				for (axis in 0...2) {
					if (control[axis]) size[axis] = canvasPlusBound(children[i], axis, slots[axis]);
					pos[axis] += canvasPlusRound((slots[axis] - size[axis]) * align[axis]);
				}
				boxes.set(children[i].id, canvasPlusCell(pos, size));
				cursor += slots[main] + spacing;
			}
			return boxes;
		}

		var preferred = new Array<Array<Int>>();
		var mins = new Array<Int>();
		var slots = new Array<Int>();
		var flex = new Array<Int>();
		var free = available[main] - spacing * (count - 1);
		for (i in 0...count) {
			var child = children[i];
			preferred.push(canvasPlusPreferredSize(child));
			mins.push(canvasPlusBound(child, main, 1));
			var stretch = control[main] && canvasPlusStretch(child, main, true);
			var slot = stretch ? mins[i] : preferred[i][main];
			slots.push(slot);
			free -= slot;
			if (force[main] || stretch) flex.push(i);
		}
		if (free < 0 && control[main]) {
			var capacity = 0;
			for (i in 0...count) capacity += slots[i] - mins[i];
			var shrink = Std.int(Math.min(capacity, -free));
			var seen = 0, used = 0;
			for (i in 0...count) {
				seen += slots[i] - mins[i];
				var target = capacity == 0 ? 0 : canvasPlusRound(shrink * seen / capacity);
				slots[i] -= target - used;
				used = target;
			}
			free += shrink;
		}
		while (flex.length > 0 && free > 0) {
			var share = Std.int(free / flex.length), remainder = free % flex.length;
			var remaining = new Array<Int>(), distributed = 0;
			for (i in 0...flex.length) {
				var index = flex[i];
				var offer = share + (i < remainder ? 1 : 0);
				var extra = canvasPlusBound(children[index], main, slots[index] + offer) - slots[index];
				slots[index] += extra;
				distributed += extra;
				if (extra == offer) remaining.push(index);
			}
			if (distributed == 0) break;
			free -= distributed;
			flex = remaining;
		}
		var cursor = canvasPlusRound(free * align[main]);
		var crossSlot = Std.int(Math.max(1, available[cross]));
		for (i in 0...count) {
			var child = children[i];
			var size = preferred[i];
			if (control[main]) size[main] = slots[i];
			if (control[cross] && (canvasPlusStretch(child, cross, true) || force[cross])) size[cross] = crossSlot;
			else if (control[cross]) size[cross] = Std.int(Math.min(size[cross], crossSlot));
			size[cross] = canvasPlusBound(child, cross, size[cross]);
			var pos = origin.copy();
			pos[main] += cursor + canvasPlusRound((slots[i] - size[main]) * align[main]);
			pos[cross] += canvasPlusRound((available[cross] - size[cross]) * align[cross]);
			boxes.set(child.id, canvasPlusCell(pos, size));
			cursor += slots[i] + spacing;
		}
		return boxes;
	}

	static function canvasPlusPreferredSize(child: TElement): Array<Int> {
		var size: Array<Int> = Reflect.field(child, "_editor_layout_preferred_size");
		if (size == null || size.length != 2) size = [child.width, child.height];
		return [canvasPlusBound(child, 0, size[0]), canvasPlusBound(child, 1, size[1])];
	}

	static inline function canvasPlusCell(pos: Array<Int>, size: Array<Int>): CanvasPlusBox
		return {x: pos[0], y: pos[1], width: size[0], height: size[1], ax: 0, ay: 0, managed: true};

	static function canvasPlusRect(x: Float, y: Float, w: Float, height: Float): CanvasPlusRect
		return {x: x, y: y, w: w, h: height};

	static function canvasPlusIntersect(a: CanvasPlusRect, b: CanvasPlusRect): CanvasPlusRect {
		if (a == null) return b;
		var x = Math.max(a.x, b.x), y = Math.max(a.y, b.y);
		return canvasPlusRect(x, y, Math.max(0, Math.min(a.x + a.w, b.x + b.w) - x),
			Math.max(0, Math.min(a.y + a.h, b.y + b.h) - y));
	}

	static inline function canvasPlusContains(rect: CanvasPlusRect, x: Float, y: Float): Bool
		return rect == null || (x >= rect.x && y >= rect.y && x < rect.x + rect.w && y < rect.y + rect.h);

	static function canvasPlusScissor(g: kha.graphics2.Graphics, rect: CanvasPlusRect) {
		if (rect == null) g.disableScissor();
		else {
			var x = Std.int(Math.max(0, Math.floor(rect.x)));
			var y = Std.int(Math.max(0, Math.floor(rect.y)));
			var right = Math.min(screenW, Math.ceil(rect.x + rect.w));
			var bottom = Math.min(screenH, Math.ceil(rect.y + rect.h));
			g.scissor(x, y, Std.int(Math.max(0, right - x)), Std.int(Math.max(0, bottom - y)));
		}
	}

	static function canvasPlusScroll(ui: Zui, element: TElement, rect: CanvasPlusRect, overlay: Bool): CanvasPlusScroll {
		var limits = canvasPlusOverflow == null ? null : canvasPlusOverflow.get(element.id);
		if (limits == null) return null;
		var scroll = canvasPlusState.scroll.get(element.id);
		if (scroll == null) {
			scroll = {offset: [0.0, 0.0], limits: [0.0, 0.0]};
			canvasPlusState.scroll.set(element.id, scroll);
		}
		var previousColor = ui.g.color;
		var width = rect.w, height = rect.h;
		var thickness = Math.min(8 * canvasPlusRenderScale, Math.min(width / 3, height / 3));
		var activeClip = canvasPlusIntersect(canvasPlusClip, rect);
		var hovered = canvasPlusContains(activeClip, ui.inputX, ui.inputY);
		for (axis in 0...2) {
			var total = axis == 0 ? width : height;
			var limit = limits[axis] * canvasPlusRenderScale;
			scroll.limits[axis] = limit;
			scroll.offset[axis] = Math.max(0, Math.min(limit, scroll.offset[axis]));
			if (limit <= 0) continue;
			var start = axis == 0 ? rect.x : rect.y;
			var track = Math.max(1, total - (limits[1 - axis] > 0 ? thickness : 0));
			var thumb = Math.min(track, Math.max(thickness, track * total / (total + limit)));
			var travel = track - thumb;
			var delta = travel * scroll.offset[axis] / limit;
			var bar = axis == 0 ? canvasPlusRect(rect.x, rect.y + height - thickness, track, thickness)
				: canvasPlusRect(rect.x + width - thickness, rect.y, thickness, track);
			var pointer = axis == 0 ? ui.inputX : ui.inputY;
			if (!overlay && travel > 0 && ui.inputEnabled && ui.inputStarted && hovered
				&& canvasPlusState.dragId == null && canvasPlusContains(bar, ui.inputX, ui.inputY)) {
				var local = pointer - start;
				canvasPlusState.dragGrab = local >= delta && local <= delta + thumb ? local - delta : thumb / 2;
				canvasPlusState.dragId = element.id;
				canvasPlusState.dragAxis = axis;
			}
			if (!overlay && travel > 0 && canvasPlusState.dragId == element.id && canvasPlusState.dragAxis == axis) {
				scroll.offset[axis] = Math.max(0, Math.min(limit, (pointer - start - canvasPlusState.dragGrab) / travel * limit));
			}
			if (overlay) {
				ui.g.color = 0xb50a0f14;
				ui.g.fillRect(bar.x, bar.y, bar.w, bar.h);
				ui.g.color = 0xff809cab;
				ui.g.fillRect(bar.x + (axis == 0 ? delta : 0), bar.y + (axis == 1 ? delta : 0),
					axis == 0 ? thumb : bar.w, axis == 1 ? thumb : bar.h);
			}
		}
		if (overlay && hovered && ui.inputEnabled && ui.inputWheelDelta != 0 && !canvasPlusWheelUsed) {
			var axis = (ui.isShiftDown || scroll.limits[1] <= 0) && scroll.limits[0] > 0 ? 0 : 1;
			var value = Math.max(0, Math.min(scroll.limits[axis],
				scroll.offset[axis] + ui.inputWheelDelta * 36 * canvasPlusRenderScale));
			if (value != scroll.offset[axis]) {
				scroll.offset[axis] = value;
				canvasPlusWheelUsed = true;
			}
		}
		ui.g.color = previousColor;
		return scroll;
	}

	static function canvasPlusGrey(color: Int): Int {
		var value = Std.int((((color >>> 16) & 255) * .2126 + ((color >>> 8) & 255) * .7152
			+ (color & 255) * .0722) * .6 + 30);
		return (color & 0xff000000) | (value << 16) | (value << 8) | value;
	}

	static function canvasPlusRadius(rect: CanvasPlusRect, radius: Float): Float
		return Math.max(0, Math.min(radius, Math.min(rect.w, rect.h) / 2));

	static function canvasPlusRoundRect(g: kha.graphics2.Graphics, rect: CanvasPlusRect,
			radius: Float, filled: Bool, strength: Float = 1) {
		var r = canvasPlusRadius(rect, radius);
		var x = rect.x, y = rect.y, right = x + rect.w, bottom = y + rect.h;
		if (r <= 0) {
			if (filled) g.fillRect(x, y, rect.w, rect.h);
			else g.drawRect(x, y, rect.w, rect.h, strength);
			return;
		}
		var centers = [[right - r, y + r], [right - r, bottom - r], [x + r, bottom - r], [x + r, y + r]];
		var points = new Array<Array<Float>>();
		for (corner in 0...4) {
			for (step in 0...9) {
				var angle = (corner - 1 + step / 8) * Math.PI / 2;
				points.push([centers[corner][0] + Math.cos(angle) * r, centers[corner][1] + Math.sin(angle) * r]);
			}
		}
		for (i in 0...points.length) {
			var a = points[i], b = points[(i + 1) % points.length];
			if (filled) g.fillTriangle(x + rect.w / 2, y + rect.h / 2, a[0], a[1], b[0], b[1]);
			else g.drawLine(a[0], a[1], b[0], b[1], strength);
		}
	}

	static function canvasPlusShape(g: kha.graphics2.Graphics, element: TElement, rect: CanvasPlusRect, radius: Float) {
		var stroke = element.strength == null ? 1 : element.strength;
		switch (element.type) {
			case Button, FRectangle, Rectangle:
				canvasPlusRoundRect(g, rect, radius, element.type != Rectangle, stroke);
			case FTriangle, Triangle, FCircle, Circle:
				var x = rect.x, y = rect.y, w = rect.w, height = rect.h;
				var points = element.type == FTriangle || element.type == Triangle
					? [[x + w / 2, y], [x + w, y + height], [x, y + height]]
					: [for (i in 0...48) [x + w / 2 + w / 2 * Math.cos(i * Math.PI / 24),
						y + height / 2 + height / 2 * Math.sin(i * Math.PI / 24)]];
				for (i in 0...points.length) {
					var a = points[i], b = points[(i + 1) % points.length];
					if (element.type == FTriangle || element.type == FCircle)
						g.fillTriangle(x + w / 2, y + height / 2, a[0], a[1], b[0], b[1]);
					else g.drawLine(a[0], a[1], b[0], b[1], stroke);
				}
			default:
		}
	}

	static function canvasPlusSlices(source: Float, destination: Float, leading: Int, trailing: Int): Array<Array<Float>> {
		var clip = Math.min(1, source / Math.max(1, leading + trailing));
		var left = leading * clip, right = trailing * clip;
		var scale = Math.min(canvasPlusRenderScale, destination / Math.max(1, left + right));
		return [[0, left, source - right, source], [0, left * scale, destination - right * scale, destination]];
	}

	static function canvasPlusImage(ui: Zui, canvas: TCanvas, element: TElement,
			rect: CanvasPlusRect, borders: Array<Int>): Bool {
		var image: kha.Image = getAsset(canvas, element.asset);
		if (image == null || isFontAsset(element.asset) || image.width <= 0 || image.height <= 0) return false;
		var x = canvasPlusSlices(image.width, rect.w, borders[0], borders[1]);
		var y = canvasPlusSlices(image.height, rect.h, borders[2], borders[3]);
		ui.g.color = element.color == null ? 0xffffffff : element.color;
		for (row in 0...3) for (column in 0...3) {
			var sw = x[0][column + 1] - x[0][column], sh = y[0][row + 1] - y[0][row];
			var dw = x[1][column + 1] - x[1][column], dh = y[1][row + 1] - y[1][row];
			if (sw > 0 && sh > 0 && dw > 0 && dh > 0)
				ui.g.drawScaledSubImage(image, x[0][column], y[0][row], sw, sh,
					rect.x + x[1][column], rect.y + y[1][row], dw, dh);
		}
		if (ui.getReleased(rect.h)) canvasPlusEvent(element);
		return true;
	}

	static function canvasPlusLines(text: String, width: Float, measure: String->Float): Array<String> {
		var result = new Array<String>();
		var whitespace = ~/\s+/g;
		for (paragraph in StringTools.replace(StringTools.replace(text, "\r\n", "\n"), "\r", "\n").split("\n")) {
			var line = "";
			for (word in whitespace.split(StringTools.trim(paragraph))) {
				if (word == "") continue;
				var candidate = line == "" ? word : line + " " + word;
				if (line != "" && measure(candidate) > width) {
					result.push(line);
					line = "";
					candidate = word;
				}
				if (measure(candidate) <= width) line = candidate;
				else {
					var chunk = "";
					var strings = new haxe.iterators.StringIteratorUnicode(word);
					for (code in strings) {
						var character = String.fromCharCode(code);
						if (chunk != "" && measure(chunk + character) > width) {
							result.push(chunk);
							chunk = "";
						}
						chunk += character;
					}
					line = chunk;
				}
			}
			result.push(line);
		}
		return result;
	}

	static function canvasPlusFontSize(ui: Zui, element: TElement): Int {
		var mode = Reflect.field(element, "font_mode");
		var fixed = Reflect.field(element, "font_size");
		var size = mode == "FIXED" ? (fixed == null ? 24 : fixed) : mode == "HEIGHT"
			? Math.max(4, Math.min(256, Math.round(element.height * .72))) : ui.t.FONT_SIZE;
		return Std.int(Math.max(1, Math.round(size * canvasPlusRenderScale)));
	}

	static function canvasPlusLabel(ui: Zui, element: TElement, rect: CanvasPlusRect,
			text: String, color: Int, align: Int, wrap: Bool, width: Float = 0) {
		if (text == null || text == "" || ui.ops.font == null) return;
		var font = ui.ops.font;
		var size = canvasPlusFontSize(ui, element);
		ui.g.font = font;
		ui.g.fontSize = size;
		ui.g.color = color;
		var explicitPadding = Reflect.field(element, "text_offset");
		var pad = (explicitPadding == null ? ui.t.TEXT_OFFSET : explicitPadding) * canvasPlusRenderScale;
		var limit = Math.max(1, (width > 0 ? Math.min(width, rect.w) : rect.w) - pad * 2);
		var measure = function(value: String): Float return font.width(size, value);
		var lines = wrap ? canvasPlusLines(text, limit, measure) : [text];
		var lineHeight = font.height(size);
		var row = Math.max(lineHeight, size);
		var free = rect.h - lineHeight - (lines.length - 1) * row;
		var vertical = Reflect.field(element, "text_vertical_align");
		var offset = vertical == "TOP" ? 0 : vertical == "BOTTOM" ? Math.max(0, free) : Math.max(0, free / 2);
		for (i in 0...lines.length) {
			var line = lines[i];
			var w = measure(line);
			var x = align == 1 ? (rect.w - w) / 2 : align == 2 ? rect.w - w - pad : pad;
			var y = offset + i * row;
			if (y < rect.h) ui.g.drawString(line, rect.x + x, rect.y + y);
		}
	}

	static inline function canvasPlusEvent(element: TElement) {
		if (element.event != null && element.event != "") events.push(element.event);
	}

	static function canvasPlusEndColor(ui: Zui, rect: CanvasPlusRect, idle: Int, hover: Int, pressed: Int): Int {
		if (!ui.enabled || !ui.inputEnabled || !canvasPlusContains(rect, ui.inputX, ui.inputY)) return idle;
		return ui.inputDown && canvasPlusContains(rect, ui.inputStartedX, ui.inputStartedY) ? pressed : hover;
	}

	static function canvasPlusClick(ui: Zui, rect: CanvasPlusRect): Bool {
		return ui.enabled && ui.inputEnabled && ui.inputReleased
			&& canvasPlusContains(rect, ui.inputX, ui.inputY)
			&& canvasPlusContains(rect, ui.inputStartedX, ui.inputStartedY);
	}

	static function canvasPlusDrawSpecial(ui: Zui, canvas: TCanvas, element: TElement,
			rect: CanvasPlusRect, data: Dynamic, disabled: Bool): Bool {
		if (data == null && element.type != Radio) return false;
		var theme = getTheme(canvas.theme);
		if (theme == null) theme = ui.t;
		var radius = canvasPlusNumber(data, "radius", 0) * canvasPlusRenderScale;
		if (canvasPlusField(data, "shadow") == true) {
			var offset: Array<Int> = canvasPlusField(data, "shadow_offset");
			if (offset != null && offset.length == 2) {
				ui.g.color = Std.int(canvasPlusNumber(data, "shadow_color", 0x59000000));
				canvasPlusShape(ui.g, element, canvasPlusRect(rect.x + offset[0] * canvasPlusRenderScale,
					rect.y + offset[1] * canvasPlusRenderScale, rect.w, rect.h), radius);
			}
		}
		var base = getColor(element.color, theme.BUTTON_COL);
		var text = getColor(element.color_text, element.type == Button ? theme.BUTTON_TEXT_COL : theme.TEXT_COL);
		var hover = getColor(element.color_hover, theme.BUTTON_HOVER_COL);
		var pressed = getColor(element.color_press, theme.BUTTON_PRESSED_COL);
		if (disabled) {
			base = canvasPlusGrey(base); text = canvasPlusGrey(text);
			hover = base; pressed = base;
		}
		var align: Int = element.alignment == null ? 0 : element.alignment;
		var variant = canvasPlusField(data, "variant");
		switch (element.type) {
			case Text:
				if (canvasPlusField(data, "wrap") == true) {
					var clip = canvasPlusIntersect(canvasPlusClip, canvasPlusIntersect(rect, canvasPlusRect(0, 0, screenW, screenH)));
					if (clip.w > 0 && clip.h > 0) {
						if (element.rotation == null || element.rotation == 0) canvasPlusScissor(ui.g, clip);
						canvasPlusLabel(ui, element, rect, getText(canvas, element), text, align, true,
							canvasPlusNumber(data, "wrap_width", 0) * canvasPlusRenderScale);
						if (element.rotation == null || element.rotation == 0) canvasPlusScissor(ui.g, canvasPlusClip);
					}
					return true;
				}
			case Image:
				var borders: Array<Int> = canvasPlusField(data, "slice");
				if (borders != null && borders.length == 4) return canvasPlusImage(ui, canvas, element, rect, borders);
			case TextInput:
				if (disabled) {
					var handle = h.nest(element.id);
					if (ui.textSelectedHandle == handle) ui.deselectText();
					if (ui.submitTextHandle == handle) ui.submitTextHandle = null;
					handle.changed = false;
					ui.g.color = canvasPlusGrey(getColor(element.color, theme.ACCENT_COL));
					ui.g.fillRect(rect.x, rect.y, rect.w, rect.h);
					var label = handle.text == "" ? getText(canvas, element) : handle.text;
					canvasPlusLabel(ui, element, rect, label, text, align, false);
					return true;
				}
			case Button:
				if (variant == "STEPPER") {
					var min = canvasPlusNumber(data, "step_min", 0), max = Math.max(min, canvasPlusNumber(data, "step_max", 100));
					var step = Math.max(.000001, canvasPlusNumber(data, "step", 1));
					var handle = h.nest(element.id);
					if (!canvasPlusState.steps.exists(element.id))
						canvasPlusState.steps.set(element.id, Math.max(min, Math.min(max, canvasPlusNumber(data, "step_value", min))));
					handle.value = canvasPlusState.steps.get(element.id);
					var edge = Math.min(rect.h, rect.w / 3);
					var minus = canvasPlusRect(rect.x, rect.y, edge, rect.h);
					var plus = canvasPlusRect(rect.x + rect.w - edge, rect.y, edge, rect.h);
					var direction = canvasPlusClick(ui, minus) ? -1 : canvasPlusClick(ui, plus) ? 1 : 0;
					var value = Math.max(min, Math.min(max, handle.value + direction * step));
					handle.changed = handle.value != value;
					if (handle.changed) {
						handle.value = value;
						canvasPlusState.steps.set(element.id, value);
						canvasPlusEvent(element);
					}
					ui.g.color = base;
					canvasPlusRoundRect(ui.g, rect, radius, true);
					var idle = Std.int(canvasPlusNumber(data, "end_color", theme.ACCENT_HOVER_COL));
					if (disabled) idle = canvasPlusGrey(idle);
					ui.g.color = canvasPlusEndColor(ui, minus, idle, hover, pressed);
					canvasPlusRoundRect(ui.g, minus, radius, true);
					ui.g.color = canvasPlusEndColor(ui, plus, idle, hover, pressed);
					canvasPlusRoundRect(ui.g, plus, radius, true);
					canvasPlusLabel(ui, element, minus, "-", text, 1, false);
					canvasPlusLabel(ui, element, plus, "+", text, 1, false);
					var precision = Math.min(6, Math.max(0, Math.ceil(-Math.log(step) / Math.log(10)) + 2));
					var divisor = Math.pow(10, precision);
					canvasPlusLabel(ui, element, canvasPlusRect(rect.x + edge, rect.y, rect.w - 2 * edge, rect.h),
						Std.string(Math.fround(handle.value * divisor) / divisor), text, align, false);
					return true;
				}
				if (radius > 0 || disabled) {
					ui.g.color = ui.getPushed(rect.h) ? pressed : ui.getHover(rect.h) ? hover : base;
					canvasPlusRoundRect(ui.g, rect, radius, true);
					canvasPlusLabel(ui, element, rect, getText(canvas, element), text, align, false);
					if (ui.getReleased(rect.h)) canvasPlusEvent(element);
					return true;
				}
			case Radio:
				var radioStyle = canvasPlusField(data, "radio_style");
				// Keep the stock inline radio behavior for legacy multi-option elements.
				if (radioStyle == null && (element.text == null || element.text.indexOf(";") == -1))
					radioStyle = "RECT";
				if (radioStyle == "RECT" || radioStyle == "DOT") {
					var selection: Map<Int, Int> = Reflect.field(canvas, "radioGroupSelection");
					if (selection == null) {
						selection = new Map<Int, Int>();
						Reflect.setField(canvas, "radioGroupSelection", selection);
					}
					var groupId = element.parent == null ? -1 : element.parent;
					var selected = selection.get(groupId) == element.id;
					if (!disabled && !selected && canvasPlusClick(ui, rect)) {
						selection.set(groupId, element.id);
						selected = true;
						canvasPlusEvent(element);
					}
					var accent = disabled ? canvasPlusGrey(theme.ACCENT_SELECT_COL) : theme.ACCENT_SELECT_COL;
					var normal = getColor(element.color, theme.ACCENT_COL);
					if (disabled) normal = canvasPlusGrey(normal);
					var labelRect = rect;
					ui.g.color = normal;
					if (radioStyle == "DOT") {
						var side = Math.min(rect.h * .66, Math.min(rect.w, 30 * canvasPlusRenderScale));
						if (side > 0) {
							var y = rect.y + (rect.h - side) / 2;
							canvasPlusRoundRect(ui.g, canvasPlusRect(rect.x, y, side, side), side / 2, false);
							if (selected) {
								ui.g.color = element.color == null ? accent : normal;
								canvasPlusRoundRect(ui.g, canvasPlusRect(rect.x + side / 4, y + side / 4,
									side / 2, side / 2), side / 4, true);
							}
							labelRect = canvasPlusRect(rect.x + side, rect.y, rect.w - side, rect.h);
						}
					}
					else {
						ui.g.color = selected && element.color == null ? accent : normal;
						ui.g.fillRect(rect.x, rect.y, rect.w, rect.h);
						if (selected && element.color != null) {
							ui.g.color = accent;
							ui.g.drawRect(rect.x, rect.y, rect.w, rect.h, 2 * canvasPlusRenderScale);
						}
					}
					canvasPlusLabel(ui, element, labelRect, getText(canvas, element), text, align, false, 0, false, ui.fontSize);
					return true;
				}
			case Check:
				if (variant == "SWITCH") {
					var handle = h.nest(element.id);
					if (!canvasPlusState.switches.exists(element.id))
						canvasPlusState.switches.set(element.id, canvasPlusField(data, "switch_value") == true);
					handle.selected = canvasPlusState.switches.get(element.id);
					handle.changed = canvasPlusClick(ui, rect);
					if (handle.changed) {
						handle.selected = !handle.selected;
						canvasPlusState.switches.set(element.id, handle.selected);
						canvasPlusEvent(element);
					}
					var track = Math.min(rect.h * .66, rect.w / 2);
					var y = rect.y + (rect.h - track) / 2;
					var color: Int = handle.selected ? theme.HIGHLIGHT_COL : theme.ACCENT_COL;
					ui.g.color = disabled ? canvasPlusGrey(color) : color;
					canvasPlusRoundRect(ui.g, canvasPlusRect(rect.x, y, track * 2, track), track / 2, true);
					var inset = track * .12, thumb = track - inset * 2;
					ui.g.color = text;
					canvasPlusRoundRect(ui.g, canvasPlusRect(rect.x + inset + (handle.selected ? track : 0), y + inset,
						thumb, thumb), thumb / 2, true);
					canvasPlusLabel(ui, element, canvasPlusRect(rect.x + track * 2, rect.y, rect.w - track * 2, rect.h),
						getText(canvas, element), text, align, false);
					return true;
				}
			case Rectangle, FRectangle:
				if (radius > 0) {
					ui.g.color = base;
					canvasPlusShape(ui.g, element, rect, radius);
					return true;
				}
			default:
		}
		return false;
	}
'''

_ARMORY_RUNTIME_TYPES = r'''
private typedef CanvasPlusBox = {
	var x: Float;
	var y: Float;
	var width: Int;
	var height: Int;
	var ax: Int;
	var ay: Int;
	var managed: Bool;
}

private typedef CanvasPlusRect = {
	var x: Float;
	var y: Float;
	var w: Float;
	var h: Float;
}

private typedef CanvasPlusScroll = {
	var offset: Array<Float>;
	var limits: Array<Float>;
}

private typedef CanvasPlusState = {
	var scroll: Map<Int, CanvasPlusScroll>;
	var steps: Map<Int, Float>;
	var switches: Map<Int, Bool>;
	var dragId: Null<Int>;
	var dragAxis: Int;
	var dragGrab: Float;
}
'''

_ARMORY_RUNTIME_MARKER = '// Armory Canvas Editor extended features v6'


def update_runtime_rotation_scope(source):
    return source.replace(
        "if (canvasPlusScrollScope) {\n\t\t\tif (rotated) ui.g.popTransformation();",
        "if (canvasPlusScrollScope) {",
    ).replace(
        "if (rotated) ui.g.pushRotation(element.rotation, ui._x + scaled(element.width) / 2, ui._y + scaled(element.height) / 2);",
        """if (rotated) {
            var pivot: Array<Float> = canvasPlusField(canvasPlusFeatures(element), "pivot");
            ui.g.pushRotation(element.rotation, ui._x + scaled(element.width) * (pivot == null ? .5 : pivot[0]),
                ui._y + scaled(element.height) * (pivot == null ? .5 : 1 - pivot[1]));
        }""",
    ).replace(
        "} else if (rotated) ui.g.popTransformation();",
        "}\n\t\tif (rotated) ui.g.popTransformation();",
    )


def build_armory_canvas_runtime(source):
    source = source.replace("return e.text;", 'return e.text == null || e.text == "" ? e.name : e.text;')
    if _ARMORY_RUNTIME_MARKER in source:
        return source
    for version, length, digest in (
        ('5', 30015, '4700031fbc485cce590b7fefdd8b2b4fe315210ea98d9025e0ac5d7a2bb3ed04'),
        ('4', 30015, '4700031fbc485cce590b7fefdd8b2b4fe315210ea98d9025e0ac5d7a2bb3ed04'),
        ('3', 30469, '0028e96d6f549cc860eca92896d2f9f1f8f274fe8bbdc2f9ecfddf8ddfdced49'),
        ('1', 28850, '05ac38e2a814eebb8e75fc3b4e7bc9c094ae44714a40e42842acb15cedae2aa7'),
        ('2', 29386, '55598b70dea5850db79ef54ed1aa700428e2cfa23a6c636aec3516c4c7edbc86'),
    ):
        old_marker = _ARMORY_RUNTIME_MARKER[:-1] + version
        if old_marker not in source:
            continue
        import hashlib
        start = source.index(old_marker) + len(old_marker) + 1
        end = start + length
        if hashlib.sha256(source[start:end].encode()).hexdigest() != digest:
            raise ValueError("The installed runtime has local edits inside its feature block")
        return source[:start].replace(old_marker, _ARMORY_RUNTIME_MARKER) + _ARMORY_RUNTIME_MEMBERS + update_runtime_rotation_scope(source[end:])
    stock = 'static function layoutGroupCells(' not in source

    def insert(anchor, content, *, first=False):
        nonlocal source
        if anchor not in source or (not first and source.count(anchor) != 1):
            raise ValueError("This Canvas.hx has a different render structure. No SDK files were edited.")
        source = source.replace(anchor, content, 1)

    insert('\t\tui.g = g;', '\t\tui.g = g;\n\t\tcanvasPlusBegin(ui, canvas, ' + str(stock).lower() + ');')
    insert('\t\tpx += anchorOffset[0];', r"""
		var canvasPlusBox = canvasPlusLayout == null ? null : canvasPlusLayout.get(element.id);
		if (canvasPlusBox != null) {
			var canvasPlusResolved = Reflect.copy(element);
			canvasPlusResolved.x = canvasPlusBox.x;
			canvasPlusResolved.y = canvasPlusBox.y;
			canvasPlusResolved.width = canvasPlusBox.width;
			canvasPlusResolved.height = canvasPlusBox.height;
			element = canvasPlusResolved;
			anchorOffset = [canvasPlusBox.ax * canvasPlusRenderScale, canvasPlusBox.ay * canvasPlusRenderScale];
		}
		px += anchorOffset[0];""".lstrip('\n'))
    insert('\t\tswitch (element.type) {', r"""
		var canvasPlusRectNow = canvasPlusRect(ui._x, ui._y, ui._w, scaled(element.height));
		var canvasPlusData = canvasPlusFeatures(element);
		var canvasPlusDisabled = canvasPlusField(canvasPlusData, "disabled") == true;
		var canvasPlusWasEnabled = ui.enabled;
		var canvasPlusSelectColor = ui.t.ACCENT_SELECT_COL;
		var canvasPlusHadInput = ui.inputEnabled;
		var canvasPlusTabPressed = ui.tabPressed;
		var canvasPlusSuppressInput = canvasPlusState != null && canvasPlusState.dragId != null
			|| !canvasPlusContains(canvasPlusClip, ui.inputX, ui.inputY)
			|| ((ui.inputDown || ui.inputReleased)
				&& !canvasPlusContains(canvasPlusClip, ui.inputStartedX, ui.inputStartedY));
		if (canvasPlusDisabled) {
			ui.enabled = false;
			if (element.type == Slider || element.type == Check || element.type == Radio) {
				var handle = h.nest(element.id);
				if (element.type == Slider) {
					if (ui.scrollHandle == handle) {
						ui.scrollHandle = null;
						ui.isScrolling = false;
					}
					if (ui.textSelectedHandle == handle) ui.deselectText();
					if (ui.submitTextHandle == handle) ui.submitTextHandle = null;
				}
				var grey = Reflect.copy(element);
				grey.color = canvasPlusGrey(getColor(element.color, ui.t.BUTTON_COL));
				grey.color_text = canvasPlusGrey(getColor(element.color_text, ui.t.TEXT_COL));
				grey.color_hover = grey.color;
				ui.t.ACCENT_SELECT_COL = canvasPlusGrey(ui.t.ACCENT_SELECT_COL);
				element = grey;
			}
		}
		if (canvasPlusSuppressInput) ui.inputEnabled = false;
		if (canvasPlusDisabled || canvasPlusSuppressInput) ui.tabPressed = false;
		var canvasPlusPreviousColor = ui.g.color;
		var canvasPlusOldHeight = ui.t.ELEMENT_H;
		var canvasPlusOldButtonHeight = ui.t.BUTTON_H;
		var canvasPlusOldOffset = ui.buttonOffsetY;
		var canvasPlusFit = canvasPlusBox != null && canvasPlusBox.managed && element.type != Text && element.type != TextArea;
		if (canvasPlusFit) {
			ui.t.ELEMENT_H = element.height;
			ui.t.BUTTON_H = element.height;
			ui.buttonOffsetY = 0;
		}
		var canvasPlusHandle = element.event != null && element.event != ""
			&& (element.type == Check || element.type == Radio || element.type == Slider)
			? h.nest(element.id) : null;
		var canvasPlusOldSelected = canvasPlusHandle == null ? false : canvasPlusHandle.selected;
		var canvasPlusOldValue = canvasPlusHandle == null ? 0.0 : canvasPlusHandle.value;
		var canvasPlusOldPosition = canvasPlusHandle == null ? 0 : canvasPlusHandle.position;
		var canvasPlusHandled = canvasPlusDrawSpecial(ui, canvas, element, canvasPlusRectNow, canvasPlusData, canvasPlusDisabled);
		if (!canvasPlusHandled) switch (element.type) {""".lstrip('\n'), first=True)
    insert('\t\tui.ops.font = font;', r"""
		if (!canvasPlusHandled && !canvasPlusDisabled && !canvasPlusSuppressInput && canvasPlusHandle != null
			&& (canvasPlusHandle.selected != canvasPlusOldSelected || canvasPlusHandle.value != canvasPlusOldValue
				|| canvasPlusHandle.position != canvasPlusOldPosition)) canvasPlusEvent(element);
		if (canvasPlusDisabled) {
			ui.enabled = canvasPlusWasEnabled;
			ui.t.ACCENT_SELECT_COL = canvasPlusSelectColor;
		}
		if (canvasPlusSuppressInput) ui.inputEnabled = canvasPlusHadInput;
		if (canvasPlusDisabled || canvasPlusSuppressInput) ui.tabPressed = canvasPlusTabPressed;
		if (canvasPlusData != null || canvasPlusHandled) ui.g.color = canvasPlusPreviousColor;
		if (canvasPlusFit) {
			ui.t.ELEMENT_H = canvasPlusOldHeight;
			ui.t.BUTTON_H = canvasPlusOldButtonHeight;
			ui.buttonOffsetY = canvasPlusOldOffset;
		}
		ui.ops.font = font;""".lstrip('\n'))
    insert('\t\tif (element.children != null) {', r"""
		var canvasPlusScrollScope = canvasPlusOverflow != null && canvasPlusOverflow.exists(element.id);
		var canvasPlusParentClip = canvasPlusClip;
		var canvasPlusCanDrawChildren = true;
		if (canvasPlusScrollScope) {
			if (rotated) ui.g.popTransformation();
			var scroll = canvasPlusScroll(ui, element, canvasPlusRectNow, false);
			px -= scroll.offset[0];
			py -= scroll.offset[1];
			canvasPlusClip = canvasPlusIntersect(canvasPlusClip,
				canvasPlusIntersect(canvasPlusRectNow, canvasPlusRect(0, 0, screenW, screenH)));
			canvasPlusCanDrawChildren = canvasPlusClip.w > 0 && canvasPlusClip.h > 0;
			if (canvasPlusCanDrawChildren) canvasPlusScissor(ui.g, canvasPlusClip);
		}
		if (canvasPlusCanDrawChildren && element.children != null) {""".lstrip('\n'), first=True)
    insert('\t\tif (rotated) ui.g.popTransformation();\n\t}', r"""
		if (canvasPlusScrollScope) {
			if (canvasPlusCanDrawChildren) {
				canvasPlusScroll(ui, element, canvasPlusRectNow, true);
				canvasPlusScissor(ui.g, canvasPlusParentClip);
			}
			canvasPlusClip = canvasPlusParentClip;
		} else if (rotated) ui.g.popTransformation();
	}""".lstrip('\n'))
    insert('class Canvas {', 'class Canvas {\n\n\t' + _ARMORY_RUNTIME_MARKER + '\n' + _ARMORY_RUNTIME_MEMBERS)
    return update_runtime_rotation_scope(source).rstrip() + '\n\n' + _ARMORY_RUNTIME_TYPES


class NODE_OT_armory_canvas_install_runtime(Operator, ImportHelper):
    bl_idname = "node.armory_canvas_install_runtime"
    bl_label = "Install Armory Runtime Features"
    bl_description = "Select Canvas.hx from the Armory SDK. A backup is written beside it"
    filter_glob: StringProperty(default='Canvas.hx', options={'HIDDEN'})

    def execute(self, context):
        path = os.path.abspath(bpy.path.abspath(self.filepath))
        if os.path.basename(path).lower() != 'canvas.hx':
            self.report({'ERROR'}, "Select armory/Sources/armory/ui/Canvas.hx in your SDK")
            return {'CANCELLED'}
        try:
            with open(path, encoding='utf-8', newline='') as source:
                original = source.read()
            normalized = original.replace('\r\n', '\n')
            updated = build_armory_canvas_runtime(normalized)
            if updated == normalized:
                self.report({'INFO'}, "Armory runtime features are already installed")
                return {'FINISHED'}
            backup = path + '.canvas-editor-v6-backup'
            if not os.path.exists(backup):
                _canvas_atomic_write(backup, original)
            else:
                with open(backup, encoding='utf-8') as source:
                    if source.read() != normalized:
                        raise ValueError("An SDK backup already exists with different contents. Reinstall on your current SDK copy.")
            if '\r\n' in original:
                updated = updated.replace('\n', '\r\n')
            _canvas_atomic_write(path, updated)
        except (ValueError, OSError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "Installed. Rebuild your Armory project to use the new canvas features.")
        return {'FINISHED'}


class NODE_PT_armory_canvas_export_and_sync(ArmoryCanvasSidebarPanelBase):
    bl_label = "Armory"
    bl_idname = "NODE_PT_armory_canvas_export_and_sync"
    bl_order = 5
    bl_options = {'DEFAULT_CLOSED'}
    def draw(self, context):
        ntree = self.get_tree_or_warn(context, self.layout)
        if not ntree:
            return
        layout = self.layout
        layout.prop(ntree, "export_filepath", text="Target")
        layout.prop(ntree, "auto_save_on_play")
        row = layout.row(align=True)
        save_label = "Export *" if (ntree.is_dirty and ntree.export_filepath) else "Export"
        op = row.operator("node.export_armory_canvas", text=save_label, icon='EXPORT')
        op.force_delay = False
        op_as = row.operator("node.export_armory_canvas", text="", icon='FILE_FOLDER')
        op_as.force_dialog = True
        layout.operator("node.import_armory_canvas", text="Load", icon='IMPORT')
        layout.operator("node.armory_canvas_generate_events", icon='FILE_SCRIPT')
        layout.operator("node.armory_canvas_install_runtime", icon='TOOL_SETTINGS')

# -------------------------------------------------------------------------
# 8. Registration
# -------------------------------------------------------------------------
classes = (
    ArmoryCanvasSizePreset,
    ArmoryCanvasTree,
    ArmoryCanvasSocket,
    ArmoryCanvasEmptyNode,
    ArmoryCanvasTextNode,
    ArmoryCanvasImageNode,
    ArmoryCanvasButtonNode,
    ArmoryCanvasCheckNode,
    ArmoryCanvasRadioNode,
    ArmoryCanvasTextInputNode,
    ArmoryCanvasTextAreaNode,
    ArmoryCanvasKeyInputNode,
    ArmoryCanvasComboBoxNode,
    ArmoryCanvasSliderNode,
    ArmoryCanvasRectNode,
    ArmoryCanvasFilledRectNode,
    ArmoryCanvasCircleNode,
    ArmoryCanvasFilledCircleNode,
    ArmoryCanvasTriangleNode,
    ArmoryCanvasFilledTriangleNode,
    ArmoryCanvasProgressNode,
    ArmoryCanvasCProgressNode,
    UI_OT_select_asset,
    NODE_OT_armory_canvas_preview_scroll,
    NODE_OT_armory_canvas_marquee,
    NODE_OT_armory_canvas_frame,
    NODE_OT_armory_canvas_align_selection,
    NODE_OT_armory_canvas_simulate,
    NODE_OT_armory_canvas_save_prefab,
    NODE_OT_armory_canvas_place_prefab,
    NODE_OT_armory_canvas_generate_events,
    NODE_OT_armory_canvas_install_runtime,
    NODE_OT_armory_canvas_gizmo,
    NODE_OT_armory_canvas_properties_clipboard,
    NODE_OT_armory_canvas_toggle_selection,
    NODE_OT_armory_canvas_toggle_group_lock,
    NODE_OT_armory_canvas_bulk_locks,
    NODE_OT_armory_canvas_delete_groups,
    NODE_MT_armory_canvas_outliner_actions,
    NODE_UL_armory_canvas_elements,
    NODE_OT_armory_canvas_fit_stretch,
    NODE_OT_armory_canvas_align_element,
    NODE_OT_armory_canvas_anchor_preset,
    NODE_PT_armory_canvas_anchor_presets,
    NODE_OT_armory_canvas_move_element,
    NODE_OT_armory_canvas_element_popup,
    NODE_MT_armory_canvas_parent,
    NODE_OT_armory_canvas_parent_menu,
    NODE_OT_armory_canvas_duplicate_hierarchy,
    NODE_OT_armory_canvas_transform_selected,
    NODE_OT_armory_canvas_parent_selected,
    NODE_OT_armory_canvas_unparent,
    NODE_OT_armory_canvas_add_element,
    NODE_OT_import_armory_canvas,
    NODE_OT_export_armory_canvas,
    ARMORY_MT_add_basic,
    ARMORY_MT_add_buttons,
    ARMORY_MT_add_inputs,
    ARMORY_MT_add_shapes,
    ARMORY_MT_add_progress,
    NODE_PT_armory_canvas_header_view,
    NODE_PT_armory_canvas_header_snapping,
    NODE_PT_armory_canvas_header_align,
    NODE_PT_armory_canvas_add_elements,
    NODE_OT_armory_canvas_apply_size,
    NODE_OT_armory_canvas_custom_size,
    NODE_OT_armory_canvas_save_size,
    NODE_OT_armory_canvas_open_preset_folder,
    NODE_OT_armory_canvas_size_picker,
    NODE_PT_armory_canvas_settings,
    NODE_OT_armory_theme_action,
    NODE_PT_armory_canvas_themes,
    NODE_PT_armory_canvas_outliner_hierarchy,
    NODE_PT_armory_canvas_selection,
    NODE_PT_armory_canvas_element_properties,
    NODE_PT_armory_canvas_viewport_and_grid,
    NODE_PT_armory_canvas_viewport_gizmos_nodes,
    NODE_PT_armory_canvas_grid_and_snapping,
    NODE_PT_armory_canvas_export_and_sync,
)

addon_keymaps = []

def draw_add_menu(self, context):
    if context.space_data.tree_type == 'ArmoryCanvasTreeType':
        layout = self.layout
        for grp_label, menu_idname, grp_icon, _grp_classes in ELEMENT_GROUPS:
            layout.menu(menu_idname, text=grp_label, icon=grp_icon)

def register():
    global _draw_handler
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.NODE_HT_header.append(draw_canvas_simulation_header)
    bpy.types.NODE_MT_add.append(draw_add_menu)

    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceNodeEditor.draw_handler_add(
            draw_armory_canvas_callback, (), 'WINDOW', 'POST_PIXEL'
        )

    if on_blend_save_pre not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(on_blend_save_pre)
    if on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)

    for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if reset_canvas_caches not in handlers:
            handlers.append(reset_canvas_caches)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        # Match Blender's keyframe-jump keymap so the canvas operator's poll
        # can take precedence here and leave animation shortcuts available elsewhere.
        frames = kc.keymaps.new(name='Frames', space_type='EMPTY', region_type='WINDOW')
        for key, shift, direction in (
            ('UP_ARROW', False, 'UP'), ('DOWN_ARROW', False, 'DOWN'),
            ('UP_ARROW', True, 'FRONT'), ('DOWN_ARROW', True, 'BACK'),
        ):
            kmi = frames.keymap_items.new("node.armory_canvas_move_element", key, 'PRESS', shift=shift, head=True)
            kmi.properties.direction = direction
            addon_keymaps.append((frames, kmi))

        km = kc.keymaps.new(name='Node Editor', space_type='NODE_EDITOR')
        for key, selected in (('HOME', False), ('NUMPAD_PERIOD', True)):
            kmi = km.keymap_items.new("node.armory_canvas_frame", key, 'PRESS', head=True)
            kmi.properties.selected = selected
            addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new("node.armory_canvas_marquee", 'B', 'PRESS', head=True)
        addon_keymaps.append((km, kmi))
        for key in ('WHEELUPMOUSE', 'WHEELDOWNMOUSE'):
            for shift in (False, True):
                kmi = km.keymap_items.new("node.armory_canvas_preview_scroll", key, 'PRESS', shift=shift, head=True)
                addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new("wm.call_panel", 'A', 'PRESS', alt=True, head=True)
        kmi.properties.name = "NODE_PT_armory_canvas_anchor_presets"
        addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new("node.armory_canvas_element_popup", 'E', 'PRESS', head=True)
        addon_keymaps.append((km, kmi))
        for shift in (False, True):
            kmi = km.keymap_items.new("node.armory_canvas_gizmo", 'LEFTMOUSE', 'PRESS', shift=shift)
            addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new("node.armory_canvas_gizmo", 'LEFTMOUSE', 'PRESS', ctrl=True, alt=True, head=True)
        addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new("node.armory_canvas_gizmo", 'LEFTMOUSE', 'DOUBLE_CLICK', head=True)
        addon_keymaps.append((km, kmi))
        for key, operation in (('G', 'MOVE'), ('R', 'ROTATE'), ('S', 'SCALE')):
            kmi = km.keymap_items.new("node.armory_canvas_transform_selected", key, 'PRESS')
            kmi.properties.operation = operation
            addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new("node.armory_canvas_parent_menu", 'P', 'PRESS', ctrl=True)
        addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new("node.armory_canvas_duplicate_hierarchy", 'D', 'PRESS', shift=True)
        addon_keymaps.append((km, kmi))

def unregister():
    _canvas_simulations.clear()
    reset_canvas_caches()
    for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if reset_canvas_caches in handlers:
            handlers.remove(reset_canvas_caches)
    global _draw_handler
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()
    _canvas_guides.clear()
    _canvas_marquees.clear()

    if on_blend_save_pre in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(on_blend_save_pre)
    if on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)

    if _draw_handler is not None:
        bpy.types.SpaceNodeEditor.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None

    bpy.types.NODE_HT_header.remove(draw_canvas_simulation_header)
    bpy.types.NODE_MT_add.remove(draw_add_menu)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'NODE_EDITOR':
                area.tag_redraw()

def armory_canvas_validate_element_types():
    """Check the element ID mapping against armory.ui.ElementType."""
    expected = {0:'Text', 1:'Image', 2:'Button', 3:'Empty', 6:'Check', 7:'Radio',
                8:'ComboBox', 9:'Slider', 10:'TextInput', 11:'KeyInput',
                12:'FilledRect', 13:'Rect', 14:'FilledCircle', 15:'Circle',
                16:'FilledTriangle', 17:'Triangle', 18:'Progress',
                19:'CProgress', 20:'TextArea'}
    assert set(TYPE_TO_NODE_BL_IDNAME) == set(expected), (set(expected) ^ set(TYPE_TO_NODE_BL_IDNAME))
    return True