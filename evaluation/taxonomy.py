"""Explicit name-based taxonomy alignment; numeric class IDs are dataset-local."""


def ordered_names(names):
    if isinstance(names, dict):
        keys = sorted(int(k) for k in names)
        if keys != list(range(len(keys))):
            raise ValueError('Class IDs must be contiguous from zero')
        return [names[k] if k in names else names[str(k)] for k in keys]
    return list(names)


def canonical(name):
    return str(name).strip().lower().replace('-', '').replace('_', '').replace(' ', '')


def binary_class(name):
    key = canonical(name)
    if key in {'normal', 'benign', 'nilm', 'superficialintermediate', 'parabasal', 'metaplastic'}:
        return 0
    if key in {'abnormal', 'dyskeratotic', 'koilocytotic', 'ascus', 'asch', 'lsil', 'hsil', 'scc'}:
        return 1
    raise ValueError(f'No reviewed binary mapping for class {name!r}')


def binary_index_map(names):
    return {i: binary_class(name) for i, name in enumerate(ordered_names(names))}


def native_index_map(source_names, target_names):
    target = {canonical(n): i for i, n in enumerate(ordered_names(target_names))}
    if len(target) != len(ordered_names(target_names)):
        raise ValueError('Duplicate canonical class names')
    try:
        return {i: target[canonical(n)] for i, n in enumerate(ordered_names(source_names))}
    except KeyError as exc:
        raise ValueError('Incompatible native vocabularies; select binary or class-agnostic evaluation explicitly') from exc
