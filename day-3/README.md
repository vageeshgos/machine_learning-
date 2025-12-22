Perfect 👍
Below is a **COMPLETE, READY-TO-USE `README.md` FILE** for GitHub.
You can **copy–paste this directly** into your repository.

No extra explanation.
No missing parts.
Clean, structured, interview-ready.

---

````md
# 📘 Data Structures & Algorithms (DSA) – Array Fundamentals

This repository documents my **DSA learning journey** with a strong focus on **arrays** and **problem-solving approaches** used in real interviews.

For every problem, I follow this structure:

- 🟥 Brute Force Approach  
- 🟨 Better Approach  
- 🟩 Optimal Approach (Interview Standard)

Each solution is written in **simple Python**, easy to understand, and beginner-friendly.

---

## 🧠 Topics Covered

- Time & Space Complexity (Big-O)
- Array Traversal
- Largest Element in an Array
- Second Largest Element in an Array
- Check if the Array is Sorted
- Remove Duplicates from Sorted Array
- Two Pointer Technique
- In-place Array Modification

---

## 🧩 Problems & Solutions

---

## 1️⃣ Largest Element in an Array

### 🔹 Problem Statement
Given an integer array `nums`, return the **largest element**.

---

### 🟥 Brute / Basic Approach
```python
def largestElement(nums):
    largest = nums[0]
    for i in range(1, len(nums)):
        if nums[i] > largest:
            largest = nums[i]
    return largest
````

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 2️⃣ Second Largest Element in an Array

### 🔹 Problem Statement

Return the **second largest unique element**.
If it does not exist, return `-1`.

---

### 🟥 Brute Force Approach (Sorting)

```python
def secondLargest(nums):
    nums = list(set(nums))
    nums.sort()
    if len(nums) < 2:
        return -1
    return nums[-2]
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(n)

---

### 🟨 Better Approach (Two Pass)

```python
def secondLargest(nums):
    largest = max(nums)
    second = -1
    for x in nums:
        if x != largest and x > second:
            second = x
    return second
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

### 🟩 Optimal Approach (One Pass)

```python
def secondLargest(nums):
    largest = -1
    second = -1
    for x in nums:
        if x > largest:
            second = largest
            largest = x
        elif x != largest and x > second:
            second = x
    return second
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 3️⃣ Check if the Array is Sorted

### 🔹 Problem Statement

Check whether the array is sorted in **non-decreasing order**.

---

### 🟩 Optimal Approach

```python
def isSorted(nums):
    for i in range(len(nums) - 1):
        if nums[i] > nums[i + 1]:
            return False
    return True
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 4️⃣ Remove Duplicates from Sorted Array

### 🔹 Problem Statement

Remove duplicates **in-place** from a sorted array and return the count of unique elements.

---

### 🟥 Brute Force Approach (Extra Array)

```python
def removeDuplicates(nums):
    temp = []
    for x in nums:
        if not temp or temp[-1] != x:
            temp.append(x)
    for i in range(len(temp)):
        nums[i] = temp[i]
    return len(temp)
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

### 🟨 Better Approach (Using Set)

```python
def removeDuplicates(nums):
    unique = sorted(set(nums))
    for i in range(len(unique)):
        nums[i] = unique[i]
    return len(unique)
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(n)

---

### 🟩 Optimal Approach (Two Pointers – In-place)

```python
def removeDuplicates(nums):
    if len(nums) == 0:
        return 0

    k = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1]:
            nums[k] = nums[i]
            k += 1

    return k
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 🧠 Key Learnings

* Brute force helps understand the problem
* Better approach improves efficiency
* Optimal approach is required for interviews
* Sorted arrays allow two-pointer techniques
* Always aim for **O(n)** time when possible

---

## 🚀 Upcoming Topics

* Move Zeros
* Left & Right Rotation of Array
* Binary Search
* Two Sum
* Hashing Basics
* Sliding Window Technique

---

## ⭐ Final Note

This repository is built for:

* Placement preparation
* Interview revision
* Strong DSA fundamentals

If this helps you, feel free to ⭐ star the repository.

Happy Coding 🚀

```

---

If you want next:
- 📂 Folder structure (`arrays/`, `solutions/`)
- 🧾 Separate README for each topic
- 🧠 Handwritten-notes style README
- 📊 Progress tracker for GitHub

Just say 👍
```
