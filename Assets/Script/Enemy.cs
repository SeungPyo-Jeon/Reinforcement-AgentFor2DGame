using System;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.Rendering.Universal;
using Unity.VisualScripting;

public class Enemy : MonoBehaviour
{
    protected BoxCollider2D m_Collider;
    protected Rigidbody2D m_RigidBody;

    [SerializeField]
    protected float m_Speed;
    [SerializeField]
    protected E_EnemyType m_type;
    [SerializeField]
    protected Vector2 m_InitPos;
    [SerializeField]
    protected LinearEnemy m_EnemyObject;
    [SerializeField]
    protected List<WayPointObject> m_WayPointList;
    protected int m_WayPointCount;
    protected bool m_Revert;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        if (null == m_Collider)
        {
            m_Collider = GetComponent<BoxCollider2D>();
        }
        if (null == m_RigidBody)
        {
            m_RigidBody = GetComponent<Rigidbody2D>();
        }
        if (null == m_EnemyObject)
        {
            if (m_type == E_EnemyType.Circular)
            {
                //m_EnemyObject = new CircularEnemy();
            }
            else if (m_type == E_EnemyType.Linear)
            {
                //m_EnemyObject = new LinearEnemy(this.GameObject.GetComponent<Transform>, m_Speed, m_WayPointList);
            }
            else if (m_type == E_EnemyType.LinearRepeat)
            {
                //m_EnemyObject = new LinearRepeatEnemy();
            }
        }
        if (null == m_WayPointList)
        {
            m_WayPointList = new List<WayPointObject>();
        }

        m_Revert = false;
        m_WayPointCount = 0;
    }
    // Update is called once per frame

    public void LinearMove()
    {
        transform.position = Vector3.MoveTowards(
            transform.position,
            m_WayPointList[m_WayPointCount].m_Position,//.transform.position,
            m_Speed * Time.deltaTime);
        if (CloseTarget(m_WayPointList[m_WayPointCount].m_Position, 0.05f))
        {
            if (0 <= m_WayPointCount && m_WayPointCount <= m_WayPointList.Count - 1)
                if (!m_Revert)
                {
                    if (m_WayPointCount < m_WayPointList.Count - 1)
                    {
                        ++m_WayPointCount;
                    }
                    else
                    {
                        m_Revert = !m_Revert;
                    }
                }
                else
                {
                    if (m_WayPointCount > 0)
                    {
                        --m_WayPointCount;
                    }
                    else
                    {
                        m_Revert = !m_Revert;
                    }
                }
        }
    }
    protected bool CloseTarget(Vector3 targetPos, float distance)
    {
        return Vector3.Distance(targetPos, transform.position) <= distance;
    }
    void FixedUpdate()
    {
        //m_EnemyObject.
        switch (m_type)
        {
            case E_EnemyType.Linear:
                LinearMove();
                break;
            case E_EnemyType.LinearRepeat:
                //LinearRepeatMove();
                break;
            case E_EnemyType.Circular:
                //CircularMove();
                break;
        }
        //Move();
    }
}
public enum E_EnemyType
{
    Linear,
    LinearRepeat,
    Circular
}

public abstract class EnemyObject
{
    public Vector2 m_InitPos;
    public float m_Speed;
    protected BoxCollider2D m_Collider;
    protected Rigidbody2D m_RigidBody;
    protected Transform transform;

    public EnemyObject(Transform transform, float speed)
    {
        m_Speed = speed;
        this.transform = transform;
    }
    public abstract void Move();

    protected bool CloseTarget(Vector3 targetPos, float distance)
    {
        return Vector3.Distance(targetPos, transform.position) <= distance;
    }
}

public class CircularEnemy
{
    public Vector2 m_Center;
    public float m_Radius;
    public float m_Degree;
    public CircularEnemy() //: base(null, 0.0f)
    {
        m_Center = new Vector2();
        m_Radius = 0f;
        m_Degree = 0f;
    }
    public void Move()
    {

    }


}
public class LinearEnemy //: EnemyObject
{
    [SerializeField]
    protected List<WayPointObject> m_WayPointList;
    protected int m_WayPointCount;
    protected bool m_Revert;
    protected List<WayPointObject> m_WayPointObject;
    public Vector2 m_InitPos;
    [SerializeField]
    public float m_Speed;
    public Transform transform;
    public LinearEnemy(Transform transform, float speed, List<WayPointObject> wayPointObject) //: base(transform, speed)
    {
        m_Speed = speed;
        this.transform = transform;
        //m_WayPointList = new List<Vector2>();
        //m_WayPointCount = 0;
        m_Revert = false;
        m_WayPointObject = wayPointObject;
    }
    public void Move()
    {
        transform.position = Vector3.MoveTowards(
            transform.position,
            m_WayPointList[m_WayPointCount].m_Position,//.transform.position,
            m_Speed * Time.deltaTime);

        if (CloseTarget(m_WayPointList[m_WayPointCount].m_Position, 0.05f))
        {
            if (0 <= m_WayPointCount && m_WayPointCount <= m_WayPointList.Count - 1)
                if (!m_Revert)
                {
                    if (m_WayPointCount < m_WayPointList.Count - 1)
                    {
                        ++m_WayPointCount;
                    }
                    else
                    {
                        m_Revert = !m_Revert;
                    }
                }
                else
                {
                    if (m_WayPointCount > 0)
                    {
                        --m_WayPointCount;
                    }
                    else
                    {
                        m_Revert = !m_Revert;
                    }
                }
        }
    }
    protected bool CloseTarget(Vector3 targetPos, float distance)
    {
        return Vector3.Distance(targetPos, transform.position) <= distance;
    }
}
public class LinearRepeatEnemy : EnemyObject
{

    protected int m_WayPointCount;
    public LinearRepeatEnemy() : base(null, 0.0f)
    {
        m_WayPointCount = 0;
    }
    public override void Move()
    {

    }
}
[System.Serializable]
public class WayPointObject
{
    public GameObject gameObject;
    public Vector2 m_Position;
    public WayPointObject(Vector2 position)
    {
        if (null == gameObject)
        {
            gameObject = new GameObject();
            gameObject.transform.position = position;
        }
        /*Vector2 refinedPosition = new Vector2( (int) position.x, (int) position.y);
        if (    m_Position.x - refinedPosition.x < 0.25f )
        {
            m_Position.x = refinedPosition.x;
        }else{
            m_Position.x = refinedPosition.x + 0.5f;
        }*/


    }
    void __initialize()
    {
        if (null == gameObject)
        {
            gameObject = new GameObject("WayPoint");
        }
    }
}

