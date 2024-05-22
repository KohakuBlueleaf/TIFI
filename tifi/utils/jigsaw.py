def jigsaw_schedule(length, group_size=7, min_group_size=3):
    assert min_group_size <= group_size / 2
    left = length % group_size
    slices1 = [slice(i, i + group_size) for i in range(0, length, group_size)]
    slices1.append(slice(length - left, length))
    slices2 = [slice(0, left)]
    slices2.extend(
        [
            slice(i + left, i + group_size + left)
            for i in range(left, length, group_size)
        ]
    )

    if left == 0:
        slices1.pop()
        slices2[0] = slice(slices2[0].start, slices2[0].stop + min_group_size)
        for s in range(1, len(slices2) - 1):
            slices2[s] = slice(
                slices2[s].start + min_group_size, slices2[s].stop + min_group_size
            )
        slices2[-1] = slice(slices2[-1].start + min_group_size, slices2[-1].stop)
        if len(slices1) == 1:
            slices2 = slices1
    elif left < min_group_size:
        addon = min_group_size - left
        slices1[0] = slice(slices1[0].start, slices1[0].stop - addon)
        for s in range(1, len(slices1) - 1):
            slices1[s] = slice(slices1[s].start - addon, slices1[s].stop - addon)
        slices1[-1] = slice(slices1[-1].start - addon, slices1[-1].stop)
        slices2[0] = slice(slices2[0].start, slices2[0].stop + addon)
        for s in range(1, len(slices2) - 1):
            slices2[s] = slice(slices2[s].start + addon, slices2[s].stop + addon)
        slices2[-1] = slice(slices2[-1].start + addon, slices2[-1].stop)
    while True:
        yield 0, slices1
        yield 1, slices2
